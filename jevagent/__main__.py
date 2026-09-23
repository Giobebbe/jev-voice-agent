"""CLI.

  python -m jevagent run  [--root PATH] [--mic NAME] [--dry-run] [--quiet] [--audio FILE]
  python -m jevagent text "open notes and create a new note"
  python -m jevagent devices
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path

from .config import ALLOWED_ROOTS, RUNS_DIR, Settings

C = {"partial": "\033[2m", "commit": "\033[37m", "fire": "\033[1;32m", "done": "\033[32m",
     "ask": "\033[33m", "error": "\033[31m", "info": "\033[36m"}
ICON = {"partial": "  …", "commit": "  ⏎", "fire": "  ⚡", "done": "  ✓", "ask": "  ?", "error": "  ✗", "info": "  ●"}


class TerminalUI:
    def __init__(self, show_partials: bool = True):
        self.t0 = time.monotonic()
        self.show_partials = show_partials
        self._partial_open = False

    def __call__(self, kind: str, msg: str) -> None:
        t = time.monotonic() - self.t0
        line = f"{C.get(kind, '')}{ICON.get(kind, '  ')} {t:6.2f}s  {msg}\033[0m"
        if kind == "partial":
            if not self.show_partials:
                return
            sys.stdout.write("\r\033[K" + line[:200])
            sys.stdout.flush()
            self._partial_open = True
            return
        if self._partial_open:
            sys.stdout.write("\r\033[K")
            self._partial_open = False
        print(line, flush=True)


def settings_from(args) -> Settings:
    s = Settings()
    if getattr(args, "root", None):
        s.root = Path(args.root).expanduser()
    if getattr(args, "mic", None):
        s.mic = args.mic
    s.dry_run = getattr(args, "dry_run", False)
    s.speak = not getattr(args, "quiet", False)
    return s


async def cmd_run(args) -> int:
    from .agent import Agent
    from .audio import FileSource, MicSource, Scribe, Speaker, decode_pcm16
    from .brain import Brain
    from .executor import Executor
    from .sandbox import Sandbox

    s = settings_from(args)
    ui = TerminalUI()
    sandbox = Sandbox(s.root)
    brain = Brain(s)
    speaker = Speaker(s)
    ex = Executor(s, sandbox)
    agent = Agent(s, brain, ex, speaker=speaker, ui=ui)
    source = FileSource(decode_pcm16(args.audio)) if args.audio else MicSource(s, speaker)
    ui("info", f"Jev voice agent | root: {sandbox.root} | "
               f"input: {'file ' + args.audio if args.audio else source.name} | "
               f"{'DRY RUN' if s.dry_run else 'live'}")
    await brain.warm()
    for phrase in ("Sorry, that didn't work.", "Screenshot saved.", "Bye!"):
        try:
            speaker.pcm(phrase)  # pre-cache common replies
        except Exception:
            pass
    ui("info", "listening... (say 'goodbye' or press Ctrl+C to stop)")
    try:
        await agent.run(Scribe(s), source)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        RUNS_DIR.mkdir(exist_ok=True)
        out = RUNS_DIR / f"run-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"
        with out.open("w") as f:
            for e in agent.events:
                f.write(json.dumps({"t": e.t - ui.t0, "kind": e.kind, **e.data}) + "\n")
        ui("info", f"log: {out}  | jev calls {brain.calls}, input tokens {brain.input_tokens}")
        await brain.close()
    return 0


async def cmd_text(args) -> int:
    from .brain import Brain

    s = settings_from(args)
    brain = Brain(s)
    await brain.warm()
    t = time.perf_counter()
    ds = await brain.decide_segment(" ".join(args.utterance))
    total = (time.perf_counter() - t) * 1000
    for d in ds:
        print(f"[{d.clause}] -> {d.call()}  tool_p={d.tool_p:.2f} conf={d.confidence():.2f} "
              f"finished={d.finished:.2f} jev={d.latency_ms:.0f}ms")
    print(f"total {total:.0f}ms")
    await brain.close()
    return 0


def cmd_devices(_args) -> int:
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"[{i}] {d['name']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="jevagent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="listen and act")
    r.add_argument("--root", default=str(ALLOWED_ROOTS[0]), help="sandbox root (Folder JEV or Folder JEV copy)")
    r.add_argument("--mic", help="input device name substring (default: MacBook Pro Microphone)")
    r.add_argument("--dry-run", action="store_true", help="decide but do not act")
    r.add_argument("--quiet", action="store_true", help="no voice replies or chimes")
    r.add_argument("--audio", help="replay an audio file instead of the microphone")
    t = sub.add_parser("text", help="decide one typed utterance")
    t.add_argument("utterance", nargs="+")
    sub.add_parser("devices", help="list microphones")
    args = p.parse_args()
    if args.cmd == "devices":
        return cmd_devices(args)
    try:
        return asyncio.run(cmd_run(args) if args.cmd == "run" else cmd_text(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
