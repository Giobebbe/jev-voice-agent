"""The fast ear: on-device Whisper (MLX, Apple GPU) re-transcribing the current utterance
every ~200 ms. Scribe gives a partial about once a second; this gives one five times as
often, so a clear command can fire while the user is still talking.

It is only trusted where the vocabulary is closed (an app, a site, a volume level, a
yes/no action): a slightly misheard word still maps to the right option through Jev.
Free text (search queries, titles, names) stays with Scribe, which hears names better.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np

from .audio import RATE

MODEL = "mlx-community/whisper-base.en-mlx"
PREROLL_S = 0.3
END_SILENCE_S = 0.35  # trailing silence that makes the local transcript "complete"
MAX_S = 10.0
EVERY_S = 0.2

# Tools whose arguments are a closed set or absent: safe to act on the fast transcript.
FAST_TOOLS = {"open_app", "quit_app", "open_website", "close_tab", "take_photo", "take_screenshot",
              "show_folder", "list_items", "set_volume", "dark_mode", "tell_time", "stop_listening",
              "open_item"}


class FastEar:
    def __init__(self, meter, on_text: Callable[[int, str, bool], None], model: str = MODEL):
        """on_text(utterance_id, text, complete) is called from the event loop."""
        self.meter = meter
        self.on_text = on_text
        self.model = model
        self._ring: list[bytes] = []  # pre-roll while idle
        self._buf: list[bytes] = []
        self._active = False
        self._utt = 0
        self._last_run = 0.0
        self._busy = False
        self._last_text = ""
        self._complete_sent = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def warm(self) -> None:
        import mlx_whisper
        mlx_whisper.transcribe(np.zeros(RATE, dtype=np.float32), path_or_hf_repo=self.model)

    def feed(self, chunk: bytes) -> None:
        """Called for every 100 ms chunk, after the speech meter has seen it."""
        now = time.monotonic()
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        if not self._active:
            self._ring.append(chunk)
            self._ring = self._ring[-int(PREROLL_S * 10):]
            if self.meter.talking:
                self._active = True
                self._utt += 1
                self._buf = list(self._ring)
                self._last_text = ""
                self._complete_sent = False
            return
        self._buf.append(chunk)
        self._buf = self._buf[-int(MAX_S * 10):]
        if self.meter.talking:
            self._complete_sent = False  # the user went on after a pause
        silent_for = now - self.meter.last_voice
        complete = not self.meter.talking and silent_for >= END_SILENCE_S
        if complete and self._complete_sent:
            if silent_for > 1.5:  # utterance over, go idle
                self._active = False
                self._ring = []
            return
        if not self._busy and (complete or now - self._last_run >= EVERY_S):
            self._last_run = now
            self._busy = True
            pcm = b"".join(self._buf)
            asyncio.ensure_future(self._transcribe(self._utt, pcm, complete))

    async def _transcribe(self, utt: int, pcm: bytes, complete: bool) -> None:
        try:
            text = await asyncio.to_thread(self._run, pcm)
        finally:
            self._busy = False
        if utt != self._utt:
            return
        if complete:
            self._complete_sent = True
        if text and (text != self._last_text or complete):
            self._last_text = text
            self.on_text(utt, text, complete)

    def _run(self, pcm: bytes) -> str:
        import mlx_whisper
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        r = mlx_whisper.transcribe(x, path_or_hf_repo=self.model, language="en", temperature=0.0,
                                   condition_on_previous_text=False, no_speech_threshold=0.6)
        return r.get("text", "").strip()
