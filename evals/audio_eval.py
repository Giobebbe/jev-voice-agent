"""Audio eval: spoken scenarios -> real Scribe realtime -> real agent -> fired actions.

Each utterance is synthesized once (cached), the scenario is streamed at real-time pace,
and every fired action is matched in order against the expected ones.
Timing is measured against the end of speech of the utterance the action belongs to:
negative = fired before the user finished talking.

File actions run for real inside the sandbox root (and what the eval created is removed
at the end); every other action is simulated.

  python evals/audio_eval.py [--scenario ID] [--root PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jevagent.agent import Agent  # noqa: E402
from jevagent.audio import RATE, FileSource, Scribe, Speaker  # noqa: E402
from jevagent.brain import Brain, Decision  # noqa: E402
from jevagent.config import ALLOWED_ROOTS, Settings  # noqa: E402
from jevagent.executor import Executor, Result  # noqa: E402
from jevagent.sandbox import Sandbox  # noqa: E402
from text_eval import arg_ok  # noqa: E402

SCEN = ROOT / "evals" / "scenarios.json"
RESULTS = ROOT / "evals" / "results"
LEAD_S, GAP_S = 1.0, 1.8
FILE_TOOLS = {"create_folder", "create_file", "delete_item", "rename_item", "move_item", "list_items"}
EARLY_TOOLS = {"open_app", "open_website", "show_folder"}


class EvalExecutor(Executor):
    """Real file operations, simulated everything else."""

    def run(self, d: Decision) -> Result:
        if d.tool in FILE_TOOLS:
            self.s.dry_run = False
            try:
                return super().run(d)
            finally:
                self.s.dry_run = True
        return super().run(d)


def speech_bounds(pcm: bytes) -> tuple[float, float]:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    frame = int(0.02 * RATE)
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.where(rms > max(300.0, rms.max() * 0.05))[0]
    if len(loud) == 0:
        return 0.0, len(x) / RATE
    return loud[0] * 0.02, (loud[-1] + 1) * 0.02


def build(scenario: dict, voices: dict, speaker: Speaker) -> tuple[bytes, list[dict]]:
    voice = voices[scenario["voice"]]
    pcm = b"\x00" * int(LEAD_S * RATE) * 2
    utts = []
    for u in scenario["utterances"]:
        clip = speaker.pcm(u["say"], rate=RATE, voice_id=voice)
        s, e = speech_bounds(clip)
        offset = len(pcm) / 2 / RATE
        utts.append({**u, "start": offset + s, "end": offset + e})
        pcm += clip + b"\x00" * int(GAP_S * RATE) * 2
    return pcm, utts


def match(fires: list[dict], utts: list[dict]) -> tuple[list[dict], list[dict]]:
    expected = [(i, e) for i, u in enumerate(utts) for e in u["expect"]]
    used: set[int] = set()
    rows = []
    cursor = 0
    for ui, e in expected:
        hit = None
        for k in range(cursor, len(fires)):
            f = fires[k]
            if k in used or f["tool"] != e["tool"]:
                continue
            if all(arg_ok(v, f["args"].get(a) or "") for a, v in e.get("args", {}).items()):
                hit = k
                break
        u = utts[ui]
        if hit is None:
            rows.append({"utt": ui, "say": u["say"], "expect": e, "ok": False})
            continue
        used.add(hit)
        cursor = hit + 1
        f = fires[hit]
        rows.append({"utt": ui, "say": u["say"], "expect": e, "ok": True, "call": f["call"],
                     "vs_end_ms": (f["t"] - u["end"]) * 1000, "mode": f["mode"]})
    false_fires = [f for k, f in enumerate(fires) if k not in used]
    return rows, false_fires


async def run_scenario(sc: dict, voices: dict, root: Path) -> dict:
    s = Settings(root=root, dry_run=True, speak=False)
    speaker = Speaker(s)
    pcm, utts = build(sc, voices, speaker)
    sandbox = Sandbox(root)
    before = set(sandbox.listing(depth=3))
    brain = Brain(s)
    await brain.warm()
    ex = EvalExecutor(s, sandbox)
    agent = Agent(s, brain, ex)
    source = FileSource(pcm, tail_secs=2.5)
    await agent.run(Scribe(s), source)
    await brain.close()
    # remove only what this eval created
    created = sorted(set(sandbox.listing(depth=3)) - before, key=len)
    for rel in created:
        try:
            if (sandbox.root / rel.rstrip("/")).exists():
                sandbox.delete(rel)
        except Exception:
            pass
    fires = []
    for ev in agent.events:
        if ev.kind != "fire":
            continue
        d = ev.data
        tool = d["call"].split("(")[0]
        args = {}
        inner = d["call"][len(tool) + 1:-1]
        for part in inner.split(", ") if inner else []:
            if "=" in part:
                k, v = part.split("=", 1)
                args[k] = v.strip("'\"")
        mode = "final" if d["final"] else "quiet" if d["quiet"] else "early"
        fires.append({"t": ev.t - source.t0, "call": d["call"], "tool": tool, "args": args, "mode": mode})
    rows, false_fires = match(fires, utts)
    transcript = [ev.data["text"] for ev in agent.events if ev.kind == "commit"]
    return {"id": sc["id"], "voice": sc["voice"], "rows": rows, "false_fires": false_fires,
            "transcript": transcript, "jev_calls": brain.calls, "errors":
            [ev.data for ev in agent.events if ev.kind == "error"],
            "exec_log": ex.log,
            "events": [{"t": round(ev.t - source.t0, 3), "kind": ev.kind, **ev.data} for ev in agent.events],
            "utterances": [{"say": u["say"], "start": round(u["start"], 3), "end": round(u["end"], 3)} for u in utts]}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--root", default=str(ALLOWED_ROOTS[0]))
    args = ap.parse_args()
    spec = json.loads(SCEN.read_text())
    scenarios = [s for s in spec["scenarios"] if not args.scenario or s["id"] == args.scenario]
    results = []
    for sc in scenarios:
        print(f"\n=== {sc['id']} ({sc['voice']}) ===", flush=True)
        r = await run_scenario(sc, spec["voices"], Path(args.root).expanduser())
        results.append(r)
        for row in r["rows"]:
            if row["ok"]:
                print(f"  ✓ {row['call']:<55} {row['mode']:<6} {row['vs_end_ms']:+6.0f}ms vs end of speech")
            else:
                print(f"  ✗ missing {row['expect']}   <- {row['say']!r}")
        for f in r["false_fires"]:
            print(f"  ! false fire {f['call']} at {f['t']:.2f}s ({f['mode']})")
        for e in r["errors"]:
            print(f"  ! error {e}")
    rows = [row for r in results for row in r["rows"]]
    ok = [row for row in rows if row["ok"]]
    ff = sum(len(r["false_fires"]) for r in results)
    early = [row for row in ok if row["expect"]["tool"] in EARLY_TOOLS]
    early_before = [row for row in early if row["vs_end_ms"] < 0]
    lat = [row["vs_end_ms"] for row in ok]
    print(f"\naudio eval: recall {len(ok)}/{len(rows)}, false fires {ff}")
    if lat:
        print(f"  action time vs end of speech: median {statistics.median(lat):+.0f}ms, "
              f"worst {max(lat):+.0f}ms")
    if early:
        print(f"  open_* fired before the sentence ended: {len(early_before)}/{len(early)} "
              f"(median {statistics.median(r['vs_end_ms'] for r in early):+.0f}ms)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"audio-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"saved {out}")
    return 0 if len(ok) == len(rows) and ff == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
