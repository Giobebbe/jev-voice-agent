"""Live eval: the video scenario and a file scenario, spoken, with REAL actions.

Everything the agent creates lands in the sandbox root. After the run the eval checks the
side effects on the Mac, then cleans up what it created (files) and what it opened
(apps it launched, browser tabs it opened). Run it from Terminal.app, which holds the
camera, microphone and screen permissions:

  scripts/in_terminal.sh python evals/live_eval.py [--root PATH] [--loopback]

--loopback plays the scenario on the speakers and listens with the real microphone
instead of streaming the audio file: the full acoustic path. Caveat measured on
2026-09-24: the MacBook's built-in echo cancellation learns the speaker signal and removes
it from the built-in mic after ~5-14 s, so only the first utterances get through (a human
voice is not affected). Use an external speaker, or a person, for a longer acoustic run.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio_eval import build, match  # noqa: E402
from jevagent.agent import Agent  # noqa: E402
from jevagent.audio import RATE, FileSource, MicSource, Scribe, Speaker  # noqa: E402
from jevagent.brain import Brain  # noqa: E402
from jevagent.config import ALLOWED_ROOTS, Settings  # noqa: E402
from jevagent.executor import Executor, osascript  # noqa: E402
from jevagent.sandbox import Sandbox  # noqa: E402

RESULTS = ROOT / "evals" / "results"
SCEN = ROOT / "evals" / "scenarios.json"


def running(app: str) -> bool:
    try:
        return osascript(f'application "{app}" is running') == "true"
    except Exception:
        return False


def chrome_urls() -> list[str]:
    if not running("Google Chrome"):
        return []
    try:
        out = osascript('tell application "Google Chrome" to get URL of every tab of every window')
    except Exception:
        return []
    return [u.strip() for u in out.replace("{", "").replace("}", "").split(",") if u.strip()]


def close_chrome_tabs(prefixes: list[str], before: list[str]) -> None:
    for p in prefixes:
        script = f'''
        tell application "Google Chrome"
            repeat with w in windows
                repeat with t in (tabs of w)
                    if (URL of t) starts with "{p}" then close t
                end repeat
            end repeat
        end tell'''
        try:
            if not any(u.startswith(p) for u in before):
                osascript(script)
        except Exception:
            pass


def check(name: str, ok: bool, detail: str = "") -> dict:
    print(f"  {'✓' if ok else '✗'} {name}{'  ' + detail if detail else ''}", flush=True)
    return {"check": name, "ok": ok, "detail": detail}


class LoopbackSource:
    """Plays the scenario on the speakers and records the real microphone."""

    def __init__(self, settings: Settings, pcm: bytes):
        self.mic = MicSource(settings)
        self.pcm = pcm
        self.t0 = 0.0

    async def chunks(self):
        import sounddevice as sd
        data = np.frombuffer(self.pcm, dtype=np.int16)
        started = threading.Event()

        def play():
            started.set()
            sd.play(data, RATE, blocking=True)

        self.t0 = time.monotonic()
        threading.Thread(target=play, daemon=True).start()
        started.wait()
        end = self.t0 + len(data) / RATE + 2.0
        async for c in self.mic.chunks():
            yield c
            if time.monotonic() > end:
                break


async def run(sc: dict, voices: dict, s: Settings, loopback: bool):
    pcm, utts = build(sc, voices, Speaker(s))
    brain = Brain(s)
    await brain.warm()
    ex = Executor(s, Sandbox(s.root))
    agent = Agent(s, brain, ex, ui=lambda k, m: k in ("fire", "done", "error", "ask", "commit") and print(f"    [{k}] {m}", flush=True))
    source = LoopbackSource(s, pcm) if loopback else FileSource(pcm, tail_secs=2.5)
    await agent.run(Scribe(s), source)
    await brain.close()
    fires = []
    for ev in agent.events:
        if ev.kind == "fire":
            call = ev.data["call"]
            tool = call.split("(")[0]
            args = {}
            inner = call[len(tool) + 1:-1]
            for part in inner.split(", ") if inner else []:
                if "=" in part:
                    k, v = part.split("=", 1)
                    args[k] = v.strip("'\"")
            fires.append({"t": ev.t - source.t0, "call": call, "tool": tool, "args": args,
                          "mode": ev.data.get("lane", "") or ("final" if ev.data["final"] else "quiet")})
    rows, false_fires = match(fires, utts)
    return agent, ex, rows, false_fires


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ALLOWED_ROOTS[0]))
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--keep", action="store_true", help="leave files and apps as they are")
    args = ap.parse_args()
    s = Settings(root=Path(args.root).expanduser(), dry_run=False, speak=False)
    sandbox = Sandbox(s.root)
    spec = json.loads(SCEN.read_text())
    by_id = {x["id"]: x for x in spec["scenarios"]}
    before_files = set(sandbox.listing(depth=3))
    before_apps = {a: running(a) for a in ("Photo Booth", "Preview", "Calculator")}
    before_tabs = chrome_urls()
    checks, report = [], {"root": str(sandbox.root), "loopback": args.loopback, "scenarios": []}

    print(f"\n=== live: video_replica ({'loopback mic' if args.loopback else 'file stream'}) ===", flush=True)
    agent, ex, rows, ff = await run(by_id["video_replica"], spec["voices"], s, args.loopback)
    for r in rows:
        print(f"  {'✓' if r['ok'] else '✗'} {r.get('call', r['expect'])} "
              f"{r.get('vs_end_ms', 0):+.0f}ms {r.get('mode', '')}")
    for f in ff:
        print(f"  ! false fire {f['call']}")
    report["scenarios"].append({"id": "video_replica", "rows": rows, "false_fires": ff, "exec": ex.log})
    time.sleep(1.5)
    notes = sorted((sandbox.root / "Notes").glob("*.txt")) if (sandbox.root / "Notes").exists() else []
    hello = [p for p in notes if p.stem.lower() == "hello"]
    checks.append(check("note file Notes/Hello.txt exists", bool(hello), ", ".join(p.name for p in notes)))
    if hello:
        first = hello[0].read_text(encoding="utf-8").split("\n")[0]
        checks.append(check("note title line says hello", first.strip().lower().strip('."') == "hello", repr(first)))
    st = ex.notes.state()
    checks.append(check("Jev Notes window open and rendering", ex.notes.visible,
                        f"last poll {time.monotonic() - ex.notes.last_seen:.1f}s ago"))
    checks.append(check("Jev Notes shows the hello note", st["current"].lower() == "hello.txt"
                        and st["content"].lower().startswith("hello"), f"{st['current']!r} {st['content'][:30]!r}"))
    urls = chrome_urls()
    checks.append(check("Chrome searched Norbert Wiener",
                        any("google.com/search" in u and "norbert" in u.lower() for u in urls)))
    checks.append(check("Chrome opened x.com", any("x.com" in u for u in urls)))
    checks.append(check("Photo Booth running", running("Photo Booth")))
    photos = sorted((sandbox.root / "Photos").glob("*.jpg")) if (sandbox.root / "Photos").exists() else []
    if photos:
        size = photos[-1].stat().st_size
        dims = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(photos[-1])],
                              capture_output=True, text=True).stdout.split()[-3::2]
        yavg = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-i", str(photos[-1]), "-vf",
                               "signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"],
                              capture_output=True, text=True).stderr.rsplit("YAVG=", 1)[-1].split()[0]
        checks.append(check("photo captured into the sandbox", dims == ["1280", "720"] and size > 3_000,
                            f"{photos[-1].name} {size} bytes {dims} brightness {yavg} (16 = dark room)"))
    else:
        checks.append(check("photo captured into the sandbox", False, "no Photos/*.jpg"))
    outside = [p for p in Path.home().joinpath("Desktop").iterdir()
               if p.name.startswith(("Hello", "Photo 20", "Untitled")) and p.parent != sandbox.root]
    checks.append(check("nothing created on the Desktop outside the sandbox", not outside, str(outside)))

    print("\n=== live: files ===", flush=True)
    agent, ex, rows, ff = await run(by_id["files"], spec["voices"], s, args.loopback)
    for r in rows:
        print(f"  {'✓' if r['ok'] else '✗'} {r.get('call', r['expect'])} {r.get('vs_end_ms', 0):+.0f}ms")
    report["scenarios"].append({"id": "files", "rows": rows, "false_fires": ff, "exec": ex.log})
    checks.append(check("file actions all succeeded", all(o == "ok" for _, o in ex.log), str(ex.log)))
    checks.append(check("Project folder created then deleted",
                        not any(p.name.startswith("Project") for p in sandbox.root.iterdir())))

    all_rows = [r for sc in report["scenarios"] for r in sc["rows"]]
    n_ok = sum(r["ok"] for r in all_rows)
    n_ff = sum(len(sc["false_fires"]) for sc in report["scenarios"])
    n_checks = sum(c["ok"] for c in checks)
    print(f"\nlive eval: actions {n_ok}/{len(all_rows)}, false fires {n_ff}, side-effect checks {n_checks}/{len(checks)}")
    report.update({"checks": checks, "actions_ok": n_ok, "actions": len(all_rows), "false_fires": n_ff})

    if not args.keep:
        created = sorted(set(sandbox.listing(depth=3)) - before_files, key=lambda r: -r.count("/"))
        ex.notes.hide()
        time.sleep(0.6)
        for rel in created:
            try:
                if (sandbox.root / rel.rstrip("/")).exists():
                    sandbox.delete(rel)
            except Exception:
                pass
        for app, was in before_apps.items():
            if not was and running(app):
                try:
                    osascript(f'tell application "{app}" to quit saving no')
                except Exception:
                    pass
        close_chrome_tabs(["https://www.google.com/search?q=Norbert", "https://x.com"], before_tabs)
        print("cleaned up: files created by the eval, apps it launched, tabs it opened")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"live-{'loop' if args.loopback else 'file'}-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"saved {out}")
    return 0 if n_ok == len(all_rows) and n_ff == 0 and n_checks == len(checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
