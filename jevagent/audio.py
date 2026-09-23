"""Audio in and out: microphone / file sources, ElevenLabs Scribe v2 Realtime, TTS.

Sources yield 100 ms chunks of 16 kHz mono PCM16 at real-time pace. While the speaker is
playing, the microphone chunks are replaced by silence so the agent never hears itself.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

import httpx
import numpy as np
import sounddevice as sd
import websockets

from .config import AUDIO_CACHE, Settings, secret

RATE = 16000
CHUNK = RATE // 10  # 100 ms
CHUNK_BYTES = CHUNK * 2
FFMPEG = "/opt/homebrew/bin/ffmpeg"


def input_device(name: str) -> int:
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower():
            return i
    names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    raise RuntimeError(f"No input device matching {name!r}. Available: {names}")


def decode_pcm16(path: Path | str) -> bytes:
    out = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "s16le",
                          "-ac", "1", "-ar", str(RATE), "-"], capture_output=True, check=True)
    return out.stdout


class Speaker:
    """ElevenLabs Flash TTS with an on-disk cache; plays on the default output."""

    OUT_RATE = 22050

    def __init__(self, settings: Settings):
        self.s = settings
        self.playing = threading.Event()
        self._lock = threading.Lock()
        AUDIO_CACHE.mkdir(exist_ok=True)

    def pcm(self, text: str, rate: int | None = None, voice_id: str | None = None) -> bytes:
        rate = rate or self.OUT_RATE
        voice_id = voice_id or self.s.voice_id
        key = hashlib.sha1(f"{voice_id}|{self.s.tts_model}|{rate}|{text}".encode()).hexdigest()[:16]
        path = AUDIO_CACHE / f"{key}.pcm"
        if path.exists():
            return path.read_bytes()
        r = httpx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
            params={"output_format": f"pcm_{rate}"},
            headers={"xi-api-key": secret("ELEVENLABS_API_KEY")},
            json={"text": text, "model_id": self.s.tts_model},
            timeout=20,
        )
        r.raise_for_status()
        path.write_bytes(r.content)
        return r.content

    def say_blocking(self, text: str) -> None:
        if not text or not self.s.speak:
            return
        data = np.frombuffer(self.pcm(text), dtype=np.int16)
        with self._lock:
            self.playing.set()
            try:
                sd.play(data, self.OUT_RATE, blocking=True)
            finally:
                time.sleep(0.15)  # room tail before the mic opens again
                self.playing.clear()

    async def say(self, text: str) -> None:
        await asyncio.to_thread(self.say_blocking, text)

    def chime(self) -> None:
        if self.s.speak:
            subprocess.Popen(["afplay", "-v", "0.35", "/System/Library/Sounds/Tink.aiff"])


class MicSource:
    def __init__(self, settings: Settings, speaker: Speaker | None = None):
        self.device = input_device(settings.mic)
        self.speaker = speaker
        self.name = sd.query_devices(self.device)["name"]

    async def chunks(self) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes] = asyncio.Queue()

        def callback(indata, frames, t, status):  # runs on the audio thread
            data = bytes(indata)
            if self.speaker is not None and self.speaker.playing.is_set():
                data = b"\x00" * len(data)
            loop.call_soon_threadsafe(q.put_nowait, data)

        with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK,
                               device=self.device, callback=callback):
            while True:
                yield await q.get()


class FileSource:
    """Replays PCM at real-time pace; `t0` is the wall time of sample 0."""

    def __init__(self, pcm: bytes, tail_secs: float = 2.0):
        self.pcm = pcm + b"\x00" * int(tail_secs * RATE * 2)
        self.t0 = 0.0

    async def chunks(self) -> AsyncIterator[bytes]:
        self.t0 = time.monotonic()
        for k, i in enumerate(range(0, len(self.pcm), CHUNK_BYTES)):
            due = self.t0 + k * 0.1
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            yield self.pcm[i:i + CHUNK_BYTES]


class Scribe:
    """ElevenLabs Scribe v2 Realtime over WebSocket, VAD commits."""

    URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    def __init__(self, settings: Settings):
        self.s = settings

    def url(self) -> str:
        params = [
            ("model_id", "scribe_v2_realtime"),
            ("audio_format", "pcm_16000"),
            ("commit_strategy", "vad"),
            ("vad_silence_threshold_secs", str(self.s.vad_silence_secs)),
            ("language_code", "en"),
        ]
        params += [("keyterms", k) for k in self.s.keyterms]
        return self.URL + "?" + urllib.parse.urlencode(params)

    async def run(self, source, on_partial: Callable[[str], None], on_commit: Callable[[str], None],
                  on_ready: Callable[[], Awaitable[None] | None] | None = None,
                  stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        headers = {"xi-api-key": secret("ELEVENLABS_API_KEY")}
        async with websockets.connect(self.url(), additional_headers=headers, max_size=2**22) as ws:
            first = json.loads(await ws.recv())
            if first.get("message_type") != "session_started":
                raise RuntimeError(f"Scribe did not start: {first}")
            if on_ready:
                r = on_ready()
                if asyncio.iscoroutine(r):
                    await r

            async def send() -> None:
                async for chunk in source.chunks():
                    if stop.is_set():
                        break
                    await ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(chunk).decode(),
                        "commit": False,
                        "sample_rate": RATE,
                    }))
                stop.set()

            async def recv() -> None:
                while not stop.is_set():
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                    except asyncio.TimeoutError:
                        continue
                    kind = msg.get("message_type")
                    if kind == "partial_transcript":
                        on_partial(msg.get("text", ""))
                    elif kind == "committed_transcript":
                        on_commit(msg.get("text", ""))
                    elif kind and ("error" in kind or kind == "auth_error"):
                        raise RuntimeError(f"Scribe error: {msg}")

            sender = asyncio.create_task(send())
            try:
                await recv()
            finally:
                sender.cancel()
