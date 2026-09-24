"""The decision layer: every judgment the agent makes is a Jev question.

Two request shapes:
  split   one Noul per candidate boundary ("and", "then") -> is the right side a new request?
  decide  one fan-out request per clause: tool Choice + every argument question at once
Answers are cached by (clause, context), so repeated partial transcripts cost nothing.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import text as T
from .config import Settings, secret
from .spec import (
    ABSENT, APP_HINTS, NOT_INSTALLED, MODE, NONE, SITES, SPAN_QUESTIONS, TOOL_QUESTION, TOOLS, VOLUME,
)

API = "https://api.typesafe.ai/v1/systemone"


@dataclass
class Arg:
    value: str | None
    p: float
    omitted: bool = False


@dataclass
class Decision:
    clause: str
    tool: str
    tool_p: float
    args: dict[str, Arg] = field(default_factory=dict)
    cut_off: float = 0.0
    addressed: float = 1.0
    latency_ms: float = 0.0
    runner_up: tuple[str, float] = ("", 0.0)

    @property
    def spec(self):
        return TOOLS[self.tool]

    def missing(self) -> list[str]:
        return [a for a in self.spec.args if self.args.get(a) is None or self.args[a].omitted]

    def confidence(self) -> float:
        ps = [self.tool_p]
        for a in self.spec.args + self.spec.optional:
            arg = self.args.get(a)
            if arg is not None and not arg.omitted:
                ps.append(arg.p)
        return min(ps)

    def call(self) -> str:
        parts = []
        for a in self.spec.args + self.spec.optional:
            arg = self.args.get(a)
            if arg is not None and not arg.omitted:
                parts.append(f"{a}={arg.value!r}")
        return f"{self.tool}({', '.join(parts)})"

    def key(self) -> tuple:
        return (self.tool,) + tuple(
            (a, (self.args[a].value or "").lower()) for a in self.spec.args + self.spec.optional
            if a in self.args and not self.args[a].omitted
        )


def installed_apps() -> list[str]:
    names: set[str] = set()
    for d in ("/Applications", "/System/Applications", "/System/Applications/Utilities"):
        p = Path(d)
        if p.exists():
            names.update(c.stem for c in p.glob("*.app"))
    names.add("Finder")  # lives in CoreServices
    names.add("Jev Notes")  # the agent's own notes window (jevagent/notes_app.py)
    return sorted(names)


class Brain:
    def __init__(self, settings: Settings, apps: list[str] | None = None):
        self.s = settings
        self.apps = apps if apps is not None else installed_apps()
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(6.0, connect=3.0),
            headers={"Authorization": f"Bearer {secret('TYPESAFE_API_KEY')}"},
            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=120),
        )
        self._cache: dict[str, object] = {}
        self.calls = 0
        self.input_tokens = 0

    async def close(self) -> None:
        await self.http.aclose()

    async def ask(self, state, questions: dict) -> tuple[dict, float]:
        body = {"model": self.s.jev_model, "state": state, "questions": questions}
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                r = await self.http.post(API, json=body)
                if r.status_code in (429, 529, 500, 502, 503):
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                self.calls += 1
                self.input_tokens += data.get("usage", {}).get("input_tokens", 0)
                return data["answers"], (time.perf_counter() - t0) * 1000
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == 2 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 422):
                    detail = e.response.text[:500] if isinstance(e, httpx.HTTPStatusError) else str(e)
                    raise RuntimeError(f"Jev request failed: {detail}") from e
                await asyncio.sleep(0.2 * (attempt + 1))
        raise RuntimeError("Jev request failed after retries")

    async def warm(self) -> None:
        await self.ask("hello", {"x": {"type": "noul", "instructions": "Is this a greeting?"}})

    # -- 1. clauses -------------------------------------------------------------------
    async def clauses(self, segment: str) -> list[str]:
        out: list[str] = []
        sents = T.sentences(segment)
        pending: list[tuple[int, str, list[T.Boundary]]] = []
        for i, s in enumerate(sents):
            bs = T.boundaries(s)
            pending.append((i, s, bs))
        todo = [(i, s, bs) for i, s, bs in pending if bs and ("split", s) not in self._cache]
        if todo:
            state = {"sentences": [s for _, s, _ in todo]}
            questions = {}
            for k, (_, s, bs) in enumerate(todo):
                for j, b in enumerate(bs):
                    questions[f"s{k}b{j}"] = {
                        "type": "noul",
                        "instructions": {
                            "before": b.left,
                            "after": b.right,
                            "question": f"In `sentences[{k}]`, is `after` its own command to the computer "
                                        "(it asks the computer to do a new action), rather than more words "
                                        "belonging to `before`?",
                        },
                    }
            answers, _ = await self.ask(state, questions)
            for k, (_, s, bs) in enumerate(todo):
                cuts = [b for j, b in enumerate(bs) if answers[f"s{k}b{j}"]["noul"] >= 0.5]
                self._cache[("split", s)] = T.split_at(s, cuts)
        for _, s, bs in pending:
            out.extend(self._cache[("split", s)] if bs else T.split_at(s, []))
        return [c for c in out if c]

    # -- 2. one decision per clause ----------------------------------------------------
    def questions_for(self, clause: str, items: list[str], folders: list[str]) -> dict:
        q: dict = {
            "tool": {
                "type": "choice",
                "instructions": TOOL_QUESTION,
                "criteria": {name: t.description for name, t in TOOLS.items()},
            },
            "addressed": {
                "type": "noul",
                "instructions": "Is the user telling the computer to do something right now in `command`, "
                                "rather than describing, explaining to an audience, suggesting what someone "
                                "could do, or talking about the past?",
            },
            "cut_off": {
                "type": "noul",
                "instructions": "Is the last phrase of `command` unfinished, so that more words are clearly "
                                "still missing at the end?",
            },
            "app": {
                "type": "choice",
                "instructions": "Which application does the user name or refer to in `command`?",
                "criteria": {**{a: APP_HINTS.get(a) for a in self.apps},
                             NOT_INSTALLED: "an application that is not in this list, or a website",
                             ABSENT: "no application is named"},
            },
        }
        site_opts = {d: SITES.get(d) for d in T.domains(clause)}
        site_opts.update({d: v for d, v in SITES.items() if d not in site_opts})
        site_opts[ABSENT] = "no website is named"
        q["site"] = {"type": "choice",
                     "instructions": "Which website does the user want to open in `command`?",
                     "criteria": site_opts}
        q["volume"] = {"type": "choice", "instructions": "What volume change does the user ask for?",
                       "criteria": {**VOLUME, ABSENT: "no volume change"}}
        q["mode"] = {"type": "choice", "instructions": "What does the user want to do with dark mode?",
                     "criteria": {**MODE, ABSENT: "dark mode is not mentioned"}}
        if items:
            q["item"] = {"type": "choice",
                         "instructions": "Which of the existing files or folders in `files` does the user mean?",
                         "criteria": {**{i: None for i in items[:250]}, ABSENT: "none of these"}}
        q["dest"] = {"type": "choice",
                     "instructions": "Into which folder from `files` does the user want to move it?",
                     "criteria": {**{f: None for f in folders[:250]}, "/": "the main working folder",
                                  ABSENT: "no destination folder is said"}}
        spans = T.candidate_spans(clause)
        if spans:
            opts = {s: None for s in spans}
            opts[ABSENT] = "the command does not say it"
            for name, question in SPAN_QUESTIONS.items():
                q[name] = {"type": "choice", "instructions": question, "criteria": opts}
        return q

    async def decide(self, clause: str, context: dict | None = None,
                     items: list[str] | None = None, folders: list[str] | None = None) -> Decision:
        context = context or {}
        items = items or []
        folders = folders or []
        key = ("decide", T.key(clause), json.dumps(context, sort_keys=True), tuple(items), tuple(folders))
        hit = self._cache.get(key)
        if isinstance(hit, Decision):
            return Decision(**{**hit.__dict__, "clause": clause, "latency_ms": 0.0})
        if hit is None:  # first asker starts the request, later askers share it
            if len(self._cache) > 4000:  # long sessions: drop settled answers, keep in-flight ones
                self._cache = {k: v for k, v in self._cache.items() if isinstance(v, asyncio.Future)}
            hit = asyncio.ensure_future(self._decide_uncached(clause, context, items, folders))
            self._cache[key] = hit
        try:
            d = await asyncio.shield(hit)
        except Exception:
            self._cache.pop(key, None)
            raise
        self._cache[key] = d
        return Decision(**{**d.__dict__, "clause": clause})

    async def _decide_uncached(self, clause: str, context: dict, items: list[str], folders: list[str]) -> Decision:
        state = {"command": clause, "context": context}
        if items:
            state["files"] = items
        answers, ms = await self.ask(state, self.questions_for(clause, items, folders))
        return self._decode(clause, answers, ms)

    def _decode(self, clause: str, answers: dict, ms: float) -> Decision:
        tool_a = answers["tool"]
        probs = sorted(tool_a["probabilities"].items(), key=lambda kv: -kv[1])
        tool, tool_p = probs[0]
        runner = probs[1] if len(probs) > 1 else ("", 0.0)
        args: dict[str, Arg] = {}
        for name in ("app", "site", "volume", "mode", "item", "dest", *SPAN_QUESTIONS):
            a = answers.get(name)
            if a is None:
                args[name] = Arg(None, 0.0, omitted=True)
                continue
            choice = a["choice"]
            p = a["probabilities"].get(choice, 0.0)
            if name in SPAN_QUESTIONS and choice != ABSENT:
                # Nested spans ("moving to Berlin" / "moving to Berlin in it") are the same
                # answer with a different boundary: count their mass as agreement.
                core = choice.lower()
                p = min(1.0, sum(q for c, q in a["probabilities"].items()
                                 if c != ABSENT and (core in c.lower() or c.lower() in core)))
            absent = choice in (ABSENT, NOT_INSTALLED)
            args[name] = Arg(None if absent else choice, p, omitted=absent)
        return Decision(clause=clause, tool=tool, tool_p=tool_p, args=args,
                        cut_off=answers["cut_off"]["noul"], addressed=answers["addressed"]["noul"],
                        latency_ms=ms, runner_up=runner)

    async def decide_segment(self, segment: str, context: dict | None = None,
                             items: list[str] | None = None, folders: list[str] | None = None) -> list[Decision]:
        # Speculative: while Jev decides where to split, it already decides every piece the
        # split could produce, so the answer is ready when the split comes back.
        spec = set()
        for s in T.sentences(segment):
            spec.update(T.all_pieces(s, T.boundaries(s)))
        pre = [asyncio.ensure_future(self.decide(p, context, items, folders)) for p in spec]
        try:
            cl = await self.clauses(segment)
            return list(await asyncio.gather(*(self.decide(c, context, items, folders) for c in cl)))
        finally:
            for f in pre:
                f.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
