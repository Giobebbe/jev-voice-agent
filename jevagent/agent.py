"""The loop between transcripts and actions.

A worker always evaluates the newest transcript (older partials are dropped), and the
policy decides per clause: fire now, wait for more words, or ask. Actions run one at a
time in order, on a separate queue, so thinking and doing overlap.

When may a clause fire?
  early   the user is still talking: only open_app/open_website/show_folder, very confident
  closed  a later clause already started, or Scribe committed the segment, or
  quiet   the user stopped talking (measured on the audio) and the latest partial arrived
          after that, so it contains the last words, and Jev says the clause is not cut off
A clause that is cut off or misses an argument at commit time is carried into the next
segment (people pause mid-command); only if nothing follows does the agent ask.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from .brain import Brain, Decision
from .config import Settings
from .executor import Executor
from .fastear import FAST_TOOLS, FastEar
from .spec import NONE

DUP_WINDOW_S = 4.0
CARRY_S = 2.5  # how long an unfinished command waits for the rest
CUT_OFF = 0.7  # Jev's 'unfinished phrase' probability above which we wait
WHAT = {"app": "which app", "item": "which file", "query": "what to search", "site": "which website",
        "name": "the name", "title": "the title", "note_text": "what to write", "dest": "which folder",
        "volume": "the volume change", "mode": "that"}


@dataclass
class Event:
    t: float
    kind: str  # partial | commit | decision | fire | done | ask | carry | error
    data: dict = field(default_factory=dict)


class SpeechMeter:
    """Is the user talking right now? RMS against an adaptive noise floor."""

    def __init__(self):
        self.floor = 60.0
        self.last_voice = 0.0
        self.talking = False

    def feed(self, chunk: bytes, now: float) -> None:
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if len(x) == 0:
            return
        rms = float(np.sqrt((x * x).mean()))
        threshold = max(250.0, 3.0 * self.floor)
        if rms > threshold:
            self.last_voice = now
            self.talking = True
        else:
            self.floor = 0.95 * self.floor + 0.05 * max(rms, 1.0)
            self.talking = False


class Agent:
    def __init__(self, settings: Settings, brain: Brain, executor: Executor, speaker=None, ui=None):
        self.s = settings
        self.brain = brain
        self.ex = executor
        self.speaker = speaker
        self.ui = ui or (lambda *_: None)
        self.events: list[Event] = []
        self.stop = asyncio.Event()  # the whole agent
        self.stt_stop = asyncio.Event()  # the audio stream only
        self.meter = SpeechMeter()
        self._seg = 0
        self._text = ""
        self._partial_at = 0.0
        self._evaluated: tuple | None = None
        self._commits: list[tuple[int, str]] = []
        self._wake = asyncio.Event()
        self._done: dict[tuple[int, int], tuple] = {}
        self._fired_keys: list[tuple[float, tuple]] = []
        self._seen: dict[tuple[int, int], tuple] = {}  # last decision key per clause (stability)
        self._carry: tuple[str, float, Decision] | None = None  # (text, deadline, decision)
        self._actions: asyncio.Queue = asyncio.Queue()
        self.fast: FastEar | None = None
        self._fast: tuple[int, str, bool] | None = None
        self._fast_evaluated: tuple[int, str, bool] | None = None
        self._fast_wake = asyncio.Event()

    # -- inputs ------------------------------------------------------------------------
    def log(self, kind: str, **data) -> None:
        self.events.append(Event(time.monotonic(), kind, data))

    def on_audio(self, chunk: bytes) -> None:
        self.meter.feed(chunk, time.monotonic())
        if self.fast is not None:
            self.fast.feed(chunk)

    def on_fast(self, utt: int, text: str, complete: bool) -> None:
        self._fast = (utt, text, complete)
        self.log("fast", utt=utt, text=text, complete=complete)
        self._fast_wake.set()

    def on_partial(self, text: str) -> None:
        text = text.strip()
        if text and text != self._text:
            self._text = text
            self._partial_at = time.monotonic()
            self.log("partial", seg=self._seg, text=text)
            self.ui("partial", text)
            self._wake.set()

    def on_commit(self, text: str) -> None:
        text = text.strip()
        self.log("commit", seg=self._seg, text=text)
        if text:
            self.ui("commit", text)
            self._commits.append((self._seg, text))
        self._seg += 1
        self._text = ""
        self._wake.set()

    def quiet(self) -> bool:
        """The user stopped talking and the newest partial already covers the last words."""
        m = self.meter
        now = time.monotonic()
        return (not m.talking and m.last_voice > 0
                and now - m.last_voice >= self.s.quiet_ms / 1000
                and self._partial_at >= m.last_voice + self.s.scribe_lag_s)

    # -- policy ------------------------------------------------------------------------
    def should_fire(self, seg: int, idx: int, d: Decision, closed: bool, final: bool) -> str:
        """'fire', 'carry' (wait for the next segment) or '' (wait)."""
        if d.tool == NONE:
            return ""
        spec = d.spec
        if d.missing() or d.cut_off >= CUT_OFF:
            return "carry" if final and d.tool_p >= 0.8 else ""
        if spec.final_only and not final:
            return ""
        if closed:
            if d.tool_p < self.s.tool_min or d.confidence() < self.s.arg_min:
                return ""
            if spec.final_only and d.confidence() < self.s.delete_arg_min:
                return "carry"
            return "fire"
        if spec.early and d.tool_p >= self.s.early_tool_min and d.confidence() >= self.s.early_arg_min:
            stable = self._seen.get((seg, idx)) == d.key()
            if stable or d.confidence() >= 0.98:
                return "fire"
        return ""

    def _duplicate(self, key: tuple) -> bool:
        now = time.monotonic()
        self._fired_keys = [(t, k) for t, k in self._fired_keys if now - t < DUP_WINDOW_S]
        return any(k == key for _, k in self._fired_keys)

    def _same_seg_key(self, seg: int, key: tuple) -> bool:
        return any(s == seg and k == key for (s, _), k in self._done.items())

    # -- worker ------------------------------------------------------------------------
    def _with_carry(self, text: str) -> str:
        if self._carry and time.monotonic() <= self._carry[1]:
            return self._carry[0] + " " + text
        return text

    async def evaluate(self, seg: int, text: str, final: bool, quiet: bool) -> None:
        t0 = time.monotonic()
        full = self._with_carry(text)
        try:
            decisions = await self.brain.decide_segment(
                full, self.ex.desk.context(), self.ex.items(), self.ex.folders())
        except Exception as e:
            self.log("error", where="brain", error=str(e))
            self.ui("error", f"brain: {e}")
            return
        if final and self._carry and full != text:
            self._carry = None  # consumed by this segment
        n = len(decisions)
        for idx, d in enumerate(decisions):
            closed = final or idx < n - 1 or quiet
            verdict = self.should_fire(seg, idx, d, closed, final)
            self.log("decision", seg=seg, idx=idx, clause=d.clause, call=d.call(), tool_p=d.tool_p,
                     conf=d.confidence(), cut_off=d.cut_off, final=final, quiet=quiet,
                     verdict=verdict, jev_ms=d.latency_ms, eval_ms=(time.monotonic() - t0) * 1000)
            self._seen[(seg, idx)] = d.key()
            if not verdict or (seg, idx) in self._done:
                continue
            if verdict == "carry":
                if idx == n - 1:  # only the tail of a segment can continue in the next one
                    self._carry = (d.clause, time.monotonic() + CARRY_S, d)
                    self.log("carry", seg=seg, idx=idx, clause=d.clause, call=d.call())
                    self.ui("ask", f"waiting for the rest of: {d.clause!r}")
                continue
            if self._same_seg_key(seg, d.key()) or self._duplicate(d.key()):
                self._done[(seg, idx)] = d.key()
                continue
            self._done[(seg, idx)] = d.key()
            self._fired_keys.append((time.monotonic(), d.key()))
            mode = "final" if final else "quiet" if quiet else "EARLY"
            self.log("fire", seg=seg, idx=idx, call=d.call(), final=final, quiet=quiet,
                     clause=d.clause, conf=d.confidence())
            self.ui("fire", f"{d.call()}  conf {d.confidence():.2f}  {mode}  "
                            f"+{(time.monotonic() - t0) * 1000:.0f}ms")
            await self._actions.put(d)

    async def evaluate_fast(self, utt: int, text: str, complete: bool) -> None:
        """Closed-vocabulary commands on the on-device transcript, stricter thresholds."""
        t0 = time.monotonic()
        try:
            decisions = await self.brain.decide_segment(
                text, self.ex.desk.context(), self.ex.items(), self.ex.folders())
        except Exception as e:
            self.log("error", where="brain-fast", error=str(e))
            return
        n = len(decisions)
        seg = ("fast", utt)
        for idx, d in enumerate(decisions):
            if d.tool not in FAST_TOOLS:
                continue
            closed = complete or idx < n - 1
            strict = d.tool_p >= self.s.fast_tool_min and d.confidence() >= self.s.fast_arg_min
            verdict = self.should_fire(seg, idx, d, closed, final=False) if strict else ""
            self.log("decision", seg=str(seg), idx=idx, clause=d.clause, call=d.call(), tool_p=d.tool_p,
                     conf=d.confidence(), cut_off=d.cut_off, final=False, quiet=complete, lane="fast",
                     verdict=verdict, jev_ms=d.latency_ms, eval_ms=(time.monotonic() - t0) * 1000)
            self._seen[(seg, idx)] = d.key()
            if verdict != "fire" or (seg, idx) in self._done:
                continue
            self._done[(seg, idx)] = d.key()
            if self._duplicate(d.key()):
                continue
            self._fired_keys.append((time.monotonic(), d.key()))
            mode = "fast" if complete else "FAST-EARLY"
            self.log("fire", seg=str(seg), idx=idx, call=d.call(), final=False, quiet=complete, lane="fast",
                     clause=d.clause, conf=d.confidence())
            self.ui("fire", f"{d.call()}  conf {d.confidence():.2f}  {mode}  "
                            f"+{(time.monotonic() - t0) * 1000:.0f}ms")
            await self._actions.put(d)

    async def _expire_carry(self) -> None:
        if not self._carry or time.monotonic() <= self._carry[1] or self.meter.talking or self._text:
            return
        clause, _, d = self._carry
        self._carry = None
        missing = d.missing()
        what = WHAT.get(missing[0], "that") if missing else "the rest"
        self.log("ask", call=d.call(), clause=clause, what=what)
        self.ui("ask", f"{d.tool}: didn't catch {what}")
        if self.speaker:
            await self.speaker.say(f"Sorry, I didn't catch {what}.")

    async def worker(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            while self._commits:
                seg, text = self._commits.pop(0)
                await self.evaluate(seg, text, final=True, quiet=False)
            await self._expire_carry()
            text, seg = self._text, self._seg
            if not text:
                continue
            state = (seg, text, self.quiet())
            if state != self._evaluated:
                self._evaluated = state
                await self.evaluate(seg, text, final=False, quiet=state[2])

    async def fast_worker(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self._fast_wake.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            self._fast_wake.clear()
            if self._fast is not None and self._fast != self._fast_evaluated:
                self._fast_evaluated = self._fast
                await self.evaluate_fast(*self._fast)

    async def doer(self) -> None:
        while not self.stop.is_set():
            try:
                d: Decision = await asyncio.wait_for(self._actions.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if self.speaker and d.tool not in ("tell_time", "list_items", "stop_listening"):
                self.speaker.chime()
            res = await asyncio.to_thread(self.ex.run, d)
            self.log("done", call=d.call(), ok=res.ok, detail=res.detail, ms=res.ms)
            self.ui("done" if res.ok else "error", f"{d.call()} -> {res.detail or 'ok'} ({res.ms:.0f}ms)")
            if res.say and self.speaker:
                await self.speaker.say(res.say)
            if d.tool == "stop_listening":
                self.stt_stop.set()
                self.stop.set()

    async def run(self, stt, source) -> None:
        if self.s.fast_lane and self.fast is None:
            self.fast = FastEar(self.meter, self.on_fast)
            await asyncio.to_thread(self.fast.warm)
        tasks = [asyncio.create_task(self.worker()), asyncio.create_task(self.fast_worker()),
                 asyncio.create_task(self.doer())]
        try:
            await stt.run(Tap(source, self.on_audio), self.on_partial, self.on_commit, stop=self.stt_stop)
            # let the last commit be decided and acted on
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and (self._commits or not self._actions.empty()):
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.8)
        finally:
            self.stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class Tap:
    """Passes audio through to Scribe while feeding the speech meter."""

    def __init__(self, source, on_chunk):
        self.source = source
        self.on_chunk = on_chunk

    async def chunks(self):
        async for c in self.source.chunks():
            self.on_chunk(c)
            yield c
