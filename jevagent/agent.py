"""The loop between transcripts and actions.

A worker always evaluates the newest transcript (older partials are dropped), and the
policy decides per clause: fire now, wait for more words, or ask. Actions run one at a
time in order, on a separate queue, so thinking and doing overlap.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .brain import Brain, Decision
from .config import Settings
from .executor import Executor
from .spec import NONE

DUP_WINDOW_S = 4.0


@dataclass
class Event:
    t: float
    kind: str  # partial | commit | decision | fire | done | ask | error
    data: dict = field(default_factory=dict)


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
        self._seg = 0
        self._text = ""
        self._changed_at = time.monotonic()
        self._evaluated: tuple[int, str, bool] | None = None
        self._commits: list[tuple[int, str]] = []
        self._wake = asyncio.Event()
        self._done: dict[tuple[int, int], tuple] = {}
        self._fired_keys: list[tuple[float, tuple]] = []
        self._seen: dict[tuple[int, int], tuple] = {}  # last decision key per clause (stability)
        self._actions: asyncio.Queue = asyncio.Queue()

    # -- transcript callbacks (from Scribe) -------------------------------------------
    def log(self, kind: str, **data) -> None:
        self.events.append(Event(time.monotonic(), kind, data))

    def on_partial(self, text: str) -> None:
        text = text.strip()
        if text and text != self._text:
            self._text = text
            self._changed_at = time.monotonic()
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

    # -- policy ------------------------------------------------------------------------
    def should_fire(self, seg: int, idx: int, d: Decision, closed: bool, final: bool) -> str:
        """'fire', 'ask' or '' (wait)."""
        if d.tool == NONE:
            return ""
        spec = d.spec
        if d.missing():
            if final and d.tool_p >= 0.85:
                return "ask"
            return ""
        if spec.final_only and not final:
            return ""
        if closed:
            if d.tool_p < self.s.tool_min or d.confidence() < self.s.arg_min:
                return ""
            if spec.final_only and d.confidence() < self.s.delete_arg_min:
                return "ask"
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

    # -- worker ------------------------------------------------------------------------
    async def evaluate(self, seg: int, text: str, final: bool, quiet: bool) -> None:
        t0 = time.monotonic()
        try:
            decisions = await self.brain.decide_segment(
                text, self.ex.desk.context(), self.ex.items(), self.ex.folders())
        except Exception as e:
            self.log("error", where="brain", error=str(e))
            self.ui("error", f"brain: {e}")
            return
        n = len(decisions)
        for idx, d in enumerate(decisions):
            closed = final or idx < n - 1 or (quiet and d.finished >= 0.6)
            verdict = self.should_fire(seg, idx, d, closed, final)
            self.log("decision", seg=seg, idx=idx, clause=d.clause, call=d.call(), tool_p=d.tool_p,
                     conf=d.confidence(), finished=d.finished, final=final, quiet=quiet,
                     verdict=verdict, jev_ms=d.latency_ms, eval_ms=(time.monotonic() - t0) * 1000)
            self._seen[(seg, idx)] = d.key()
            if not verdict or (seg, idx) in self._done:
                continue
            if verdict == "fire" and (self._same_seg_key(seg, d.key()) or self._duplicate(d.key())):
                self._done[(seg, idx)] = d.key()
                continue
            self._done[(seg, idx)] = d.key()
            if verdict == "ask":
                missing = d.missing()
                what = {"app": "which app", "item": "which file", "query": "what to search",
                        "site": "which website", "name": "the name", "title": "the title",
                        "note_text": "what to write", "dest": "which folder"}.get(missing[0] if missing else "", "that")
                self.log("ask", seg=seg, idx=idx, call=d.call(), what=what)
                self.ui("ask", f"{d.tool}: didn't catch {what}")
                if self.speaker:
                    asyncio.create_task(self.speaker.say(f"Sorry, I didn't catch {what}."))
                continue
            self._fired_keys.append((time.monotonic(), d.key()))
            self.log("fire", seg=seg, idx=idx, call=d.call(), final=final, quiet=quiet,
                     clause=d.clause, conf=d.confidence())
            self.ui("fire", f"{d.call()}  conf {d.confidence():.2f}  "
                            f"{'final' if final else 'quiet' if quiet else 'EARLY'}  "
                            f"+{(time.monotonic() - t0) * 1000:.0f}ms")
            await self._actions.put(d)

    def _same_seg_key(self, seg: int, key: tuple) -> bool:
        return any(s == seg and k == key for (s, _), k in self._done.items())

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
            text, seg = self._text, self._seg
            if not text:
                continue
            quiet = (time.monotonic() - self._changed_at) * 1000 >= self.s.quiet_ms
            state = (seg, text, quiet)
            if state != self._evaluated:
                self._evaluated = state
                await self.evaluate(seg, text, final=False, quiet=quiet)

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
        tasks = [asyncio.create_task(self.worker()), asyncio.create_task(self.doer())]
        try:
            await stt.run(source, self.on_partial, self.on_commit, stop=self.stt_stop)
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
