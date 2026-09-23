"""Text eval: typed utterances -> what the agent would fire, compared with expectations.

Runs the real Brain (Jev) and the real Agent policy, without audio and without acting.
  python evals/text_eval.py [--only KIND] [--repeat N]
Exit code 1 if accuracy < target or any negative case fires.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jevagent.agent import Agent  # noqa: E402
from jevagent.brain import Brain  # noqa: E402
from jevagent.config import Settings  # noqa: E402

CASES = ROOT / "evals" / "cases.jsonl"
RESULTS = ROOT / "evals" / "results"
FILES = ["Notes/", "Notes/hello.txt", "Old stuff/", "Photos/", "Photos/Photo 1.jpg", "Projects/",
         "Projects/budget.txt", "ideas.md", "shopping list.txt"]
FOLDERS = [f for f in FILES if f.endswith("/")]
TARGET = 0.95


def norm(s: str) -> str:
    return re.sub(r"[^\w\s./]", "", (s or "").lower()).strip()


def arg_ok(expected, got) -> bool:
    alts = expected if isinstance(expected, list) else [expected]
    return norm(got) in {norm(a) for a in alts}


def matches(expect: list[dict], fired: list) -> bool:
    if len(expect) != len(fired):
        return False
    for e, d in zip(expect, fired):
        if e["tool"] != d.tool:
            return False
        for k, v in e.get("args", {}).items():
            a = d.args.get(k)
            if a is None or a.omitted or not arg_ok(v, a.value):
                return False
    return True


class _NullEx:
    class desk:  # noqa: N801
        @staticmethod
        def context():
            return {}


async def run_case(brain: Brain, settings: Settings, case: dict) -> dict:
    items = FILES if case.get("files") else []
    folders = FOLDERS if case.get("files") else []
    ctx = case.get("ctx", {})
    t0 = time.perf_counter()
    decisions = await brain.decide_segment(case["say"], ctx, items, folders)
    ms = (time.perf_counter() - t0) * 1000
    agent = Agent(settings, brain, _NullEx())
    partial = case.get("partial", False)
    fired = []
    n = len(decisions)
    for idx, d in enumerate(decisions):
        closed = (not partial) or idx < n - 1
        if partial and not closed:
            agent._seen[(0, idx)] = d.key()  # as if the same partial was seen twice
        verdict = agent.should_fire(0, idx, d, closed=closed, final=not partial)
        if verdict == "fire":
            fired.append(d)
    ok = matches(case["expect"], fired) or bool(case.get("accept_none") and not fired)
    return {
        "id": case["id"], "kind": case["kind"], "say": case["say"], "ok": ok, "ms": ms,
        "fired": [d.call() for d in fired],
        "decisions": [{"clause": d.clause, "call": d.call(), "tool_p": round(d.tool_p, 3),
                       "conf": round(d.confidence(), 3), "finished": round(d.finished, 3),
                       "runner_up": d.runner_up} for d in decisions],
        "expect": case["expect"],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["kind"].startswith(args.only) or c["id"] == args.only]
    settings = Settings()
    results = []
    for rep in range(args.repeat):
        brain = Brain(settings)  # fresh cache per repeat
        await brain.warm()
        sem = asyncio.Semaphore(args.concurrency)

        async def one(c):
            async with sem:
                return await run_case(brain, settings, c)

        results += await asyncio.gather(*(one(c) for c in cases))
        await brain.close()

    n = len(results)
    ok = sum(r["ok"] for r in results)
    neg = [r for r in results if r["kind"] in ("negative", "prefix")]
    false_fires = [r for r in neg if r["fired"]]
    lat = sorted(r["ms"] for r in results)
    by_kind: dict[str, list[bool]] = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["ok"])
    print(f"\ntext eval: {ok}/{n} = {ok / n:.1%}   (target {TARGET:.0%})")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:<13} {sum(v)}/{len(v)}")
    print(f"false fires on negatives/prefixes: {len(false_fires)}")
    print(f"latency per utterance: p50 {statistics.median(lat):.0f}ms  p95 {lat[int(0.95 * (n - 1))]:.0f}ms")
    fails = [r for r in results if not r["ok"]]
    for r in fails:
        print(f"\n  FAIL {r['id']} [{r['kind']}] {r['say']!r}")
        print(f"       expected {r['expect']}")
        print(f"       fired    {r['fired']}")
        for d in r["decisions"]:
            print(f"       - [{d['clause']}] {d['call']} tool_p={d['tool_p']} conf={d['conf']} "
                  f"fin={d['finished']} runner={d['runner_up']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"text-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"accuracy": ok / n, "n": n, "false_fires": len(false_fires),
                               "p50_ms": statistics.median(lat), "results": results}, indent=1))
    print(f"\nsaved {out}")
    return 0 if ok / n >= TARGET and not false_fires else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
