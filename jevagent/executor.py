"""Carries out decisions on the Mac. Knows nothing about language.

Filesystem writes go through Sandbox. Everything else is `open`, `osascript`,
`screencapture` or ffmpeg. In dry-run mode actions are recorded, not performed.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from .brain import Decision
from .config import PROTECTED_APPS, Settings
from .sandbox import Sandbox, SandboxError

FFMPEG = "/opt/homebrew/bin/ffmpeg"


@dataclass
class Result:
    ok: bool
    say: str = ""  # short spoken reply ("" = stay quiet)
    detail: str = ""
    ms: float = 0.0


@dataclass
class DeskState:
    """What the agent knows about the desktop, sent to Jev as context."""
    front_app: str = ""
    current_note: str = ""  # path relative to the sandbox root
    page: str = ""  # site open in the browser, e.g. youtube.com
    history: list[str] = field(default_factory=list)

    def context(self) -> dict:
        ctx = {}
        if self.front_app:
            ctx["front_app"] = self.front_app
        if self.current_note:
            ctx["current_note"] = Path(self.current_note).name
        if self.page and self.front_app in ("Google Chrome", "Safari"):
            ctx["browser_page"] = self.page
        return ctx


def _run(cmd: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def osascript(script: str, timeout: float = 8.0) -> str:
    out = _run(["osascript", "-e", script], timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "osascript failed")
    return out.stdout.strip()


def _as_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Executor:
    def __init__(self, settings: Settings, sandbox: Sandbox):
        self.s = settings
        self.sb = sandbox
        self.desk = DeskState()
        self.log: list[tuple[str, str]] = []  # (call, outcome) -- what evals read

    def items(self) -> list[str]:
        return self.sb.listing()

    def folders(self) -> list[str]:
        return self.sb.folders()

    def run(self, d: Decision) -> Result:
        t0 = time.perf_counter()
        call = d.call()
        if self.s.dry_run:
            res = self._simulate(d)
        else:
            try:
                res = getattr(self, "do_" + d.tool)(d)
            except SandboxError as e:
                res = Result(False, "I can only touch files inside the JEV folder.", f"sandbox: {e}")
            except Exception as e:  # an action failing must not kill the agent
                res = Result(False, "Sorry, that didn't work.", f"{type(e).__name__}: {e}")
        res.ms = (time.perf_counter() - t0) * 1000
        self.log.append((call, "ok" if res.ok else f"fail: {res.detail}"))
        self.desk.history.append(call)
        return res

    # -- dry run keeps the desk state realistic for multi-step scenarios --------------
    def _simulate(self, d: Decision) -> Result:
        a = {k: v.value for k, v in d.args.items() if not v.omitted}
        if d.tool == "open_app":
            self.desk.front_app = a.get("app", "")
        elif d.tool in ("web_search", "youtube_search", "open_website"):
            self.desk.front_app = self.s.browser
            self.desk.page = {"web_search": "google.com", "youtube_search": "youtube.com"}.get(d.tool, a.get("site", ""))
        elif d.tool == "create_note":
            self.desk.front_app = "TextEdit"
            self.desk.current_note = f"Notes/{a.get('title') or 'Untitled'}.txt"
        elif d.tool == "take_photo":
            self.desk.front_app = "Preview"
        return Result(True, detail="dry-run")

    # -- apps --------------------------------------------------------------------------
    def do_none(self, d: Decision) -> Result:
        return Result(True)

    def do_open_app(self, d: Decision) -> Result:
        app = d.args["app"].value
        out = _run(["open", "-a", app])
        if out.returncode != 0:
            return Result(False, f"I couldn't open {app}.", out.stderr.strip())
        self.desk.front_app = app
        return Result(True, detail=app)

    def do_quit_app(self, d: Decision) -> Result:
        app = d.args["app"].value
        if app in PROTECTED_APPS:
            return Result(False, f"I'll leave {app} open.", "protected app")
        osascript(f"tell application {_as_str(app)} to quit")
        if self.desk.front_app == app:
            self.desk.front_app = ""
        return Result(True, detail=app)

    # -- notes: text files in <root>/Notes, shown in TextEdit --------------------------
    def _open_in_textedit(self, path: Path) -> None:
        _run(["open", "-a", "TextEdit", str(path)])
        self.desk.front_app = "TextEdit"

    def _note_path(self) -> Path | None:
        if not self.desk.current_note:
            return None
        try:
            p = self.sb.path(self.desk.current_note)
        except SandboxError:
            return None
        return p if p.exists() else None

    def _textedit_sync(self, path: Path, content: str) -> None:
        """Update the note: through TextEdit when it is open there (so the window changes
        live and TextEdit does not see a foreign edit), otherwise on disk and then open it."""
        script = f'''
        if application "TextEdit" is running then
            tell application "TextEdit"
                repeat with doc in documents
                    try
                        if (path of doc) is {_as_str(str(path))} then
                            set text of doc to {_as_str(content)}
                            save doc
                            activate
                            return "yes"
                        end if
                    end try
                end repeat
            end tell
        end if
        return "no"'''
        if osascript(script) != "yes":
            path.write_text(content, encoding="utf-8")
            self._open_in_textedit(path)

    def do_create_note(self, d: Decision) -> Result:
        title = d.args["title"].value if not d.args["title"].omitted else ""
        self.sb.ensure_dir("Notes")
        path = self.sb.create_file(title or "Untitled", text=(title + "\n\n") if title else "", parent_rel="Notes")
        self.desk.current_note = self.sb.rel(path)
        self._open_in_textedit(path)
        return Result(True, detail=self.desk.current_note)

    def do_set_note_title(self, d: Decision) -> Result:
        title = d.args["title"].value
        path = self._note_path()
        if path is None:
            self.do_create_note(d)
            return Result(True, detail=f"new note {self.desk.current_note}")
        # A note is "title, blank line, body"; an Untitled note has no title line yet.
        old = path.read_text(encoding="utf-8")
        lines = old.split("\n")
        has_title = not path.stem.startswith("Untitled") and bool(lines[0].strip())
        body = ("\n".join(lines[1:]) if has_title else old).strip("\n")
        self._textedit_sync(path, title + "\n\n" + (body + "\n" if body else ""))
        # Keep the file name in step with the title (TextEdit follows the rename).
        new = self.sb.rename(self.sb.rel(path), title)
        self.desk.current_note = self.sb.rel(new)
        return Result(True, detail=self.desk.current_note)

    def do_write_in_note(self, d: Decision) -> Result:
        text = d.args["note_text"].value
        path = self._note_path()
        if path is None:
            self.sb.ensure_dir("Notes")
            path = self.sb.create_file("Untitled", parent_rel="Notes")
            self.desk.current_note = self.sb.rel(path)
        content = path.read_text(encoding="utf-8")
        content = (content.rstrip("\n") + "\n" + text + "\n") if content.strip() else text + "\n"
        self._textedit_sync(path, content)
        return Result(True, detail=self.desk.current_note)

    # -- web -----------------------------------------------------------------------------
    def _open_url(self, url: str) -> None:
        _run(["open", "-a", self.s.browser, url])
        self.desk.front_app = self.s.browser
        host = urllib.parse.urlparse(url).netloc
        self.desk.page = host[4:] if host.startswith("www.") else host

    def do_web_search(self, d: Decision) -> Result:
        q = d.args["query"].value
        self._open_url("https://www.google.com/search?q=" + urllib.parse.quote_plus(q))
        return Result(True, detail=q)

    def do_youtube_search(self, d: Decision) -> Result:
        q = d.args["query"].value
        self._open_url("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q))
        return Result(True, detail=q)

    def do_open_website(self, d: Decision) -> Result:
        site = d.args["site"].value
        self._open_url("https://" + site)
        return Result(True, detail=site)

    def do_close_tab(self, d: Decision) -> Result:
        if self.s.browser == "Google Chrome":
            osascript('tell application "Google Chrome" to if (count of windows) > 0 then close active tab of front window')
        else:
            osascript('tell application "Safari" to if (count of windows) > 0 then close current tab of front window')
        return Result(True)

    # -- camera and screen ----------------------------------------------------------------
    def do_take_photo(self, d: Decision) -> Result:
        photos = self.sb.ensure_dir("Photos")
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        target = self.sb.unique(photos, f"Photo {stamp}.jpg")
        self.sb.path(self.sb.rel(target))
        # Grab ~1.2 s so auto-exposure settles, keep the last frame.
        out = _run([FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-framerate", "30",
                    "-video_size", "1280x720", "-i", f"{self.s.camera}:none", "-t", "1.2",
                    "-update", "1", "-q:v", "2", "-y", str(target)], timeout=15)
        if out.returncode != 0 or not target.exists() or target.stat().st_size < 10_000:
            return Result(False, "I couldn't use the camera.", out.stderr.strip()[-300:])
        _run(["open", "-a", "Preview", str(target)])
        self.desk.front_app = "Preview"
        return Result(True, detail=self.sb.rel(target))

    def do_take_screenshot(self, d: Decision) -> Result:
        shots = self.sb.ensure_dir("Screenshots")
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        target = self.sb.unique(shots, f"Screenshot {stamp}.png")
        self.sb.path(self.sb.rel(target))
        out = _run(["screencapture", "-x", str(target)])
        if out.returncode != 0 or not target.exists():
            return Result(False, "I couldn't take the screenshot.", out.stderr.strip())
        return Result(True, "Screenshot saved.", self.sb.rel(target))

    # -- files ---------------------------------------------------------------------------
    def do_create_folder(self, d: Decision) -> Result:
        p = self.sb.create_folder(d.args["name"].value)
        return Result(True, detail=self.sb.rel(p))

    def do_create_file(self, d: Decision) -> Result:
        p = self.sb.create_file(d.args["name"].value)
        return Result(True, detail=self.sb.rel(p))

    def do_delete_item(self, d: Decision) -> Result:
        rel = d.args["item"].value
        p = self.sb.delete(rel)
        if self.desk.current_note and self.sb.root / self.desk.current_note == p:
            self.desk.current_note = ""
        return Result(True, f"Deleted {p.name}.", rel)

    def do_rename_item(self, d: Decision) -> Result:
        rel = d.args["item"].value
        p = self.sb.rename(rel, d.args["name"].value)
        if self.desk.current_note == rel:
            self.desk.current_note = self.sb.rel(p)
        return Result(True, detail=f"{rel} -> {self.sb.rel(p)}")

    def do_move_item(self, d: Decision) -> Result:
        rel, dest = d.args["item"].value, d.args["dest"].value
        p = self.sb.move(rel, dest)
        return Result(True, detail=f"{rel} -> {self.sb.rel(p)}")

    def do_open_item(self, d: Decision) -> Result:
        p = self.sb.path(d.args["item"].value.rstrip("/"))
        _run(["open", str(p)])
        return Result(True, detail=self.sb.rel(p))

    def do_show_folder(self, d: Decision) -> Result:
        _run(["open", str(self.sb.root)])
        self.desk.front_app = "Finder"
        return Result(True)

    def do_list_items(self, d: Decision) -> Result:
        top = [i for i in self.sb.listing(depth=1)]
        if not top:
            return Result(True, "The folder is empty.")
        names = ", ".join(i.rstrip("/") for i in top[:8])
        more = f", and {len(top) - 8} more" if len(top) > 8 else ""
        return Result(True, f"There {'is' if len(top) == 1 else 'are'} {len(top)}: {names}{more}.")

    # -- system --------------------------------------------------------------------------
    def do_set_volume(self, d: Decision) -> Result:
        v = d.args["volume"].value
        scripts = {
            "up": "set volume output volume ((output volume of (get volume settings)) + 15)",
            "down": "set volume output volume ((output volume of (get volume settings)) - 15)",
            "mute": "set volume output muted true",
            "unmute": "set volume output muted false",
            "max": "set volume output volume 100",
            "half": "set volume output volume 50",
        }
        osascript(scripts[v])
        return Result(True, detail=v)

    def do_dark_mode(self, d: Decision) -> Result:
        m = d.args["mode"].value
        value = {"on": "true", "off": "false", "toggle": "not dark mode"}[m]
        osascript(f'tell application "System Events" to tell appearance preferences to set dark mode to {value}')
        return Result(True, detail=m)

    def do_tell_time(self, d: Decision) -> Result:
        now = dt.datetime.now()
        return Result(True, now.strftime("It's %-I:%M %p, %A %B %-d."))

    def do_stop_listening(self, d: Decision) -> Result:
        return Result(True, "Bye!", "stop")

