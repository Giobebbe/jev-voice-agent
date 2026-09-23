"""Jev Notes: a tiny notes app the agent writes into.

The notes are plain .txt files in <root>/Notes, written only by the agent (through the
Sandbox). A local page shows them live in a Chrome app window, polling every 250 ms, so
no AppleScript is involved and the user's own Apple Notes never appear on screen.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Jev Notes</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fbfaf7;--side:#f1efe9;--ink:#1d1d1f;--mute:#8a877f;--line:#e4e1d9;--accent:#e2a03f}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--side:#232326;--ink:#f2f2f2;--mute:#8e8e93;--line:#333336;--accent:#f0b84f}}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
font:16px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif}
.app{display:grid;grid-template-columns:240px 1fr;height:100vh}
aside{background:var(--side);border-right:1px solid var(--line);overflow:auto;padding:14px 10px}
aside h1{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:4px 8px 12px}
.n{padding:10px 10px;border-radius:8px;margin-bottom:4px}
.n b{display:block;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.n span{font-size:12px;color:var(--mute)}
.n.on{background:var(--accent);color:#1d1d1f}.n.on span{color:#3b3222}
main{padding:44px 56px;overflow:auto}
.date{color:var(--mute);font-size:13px;margin-bottom:14px}
h2{font-size:34px;line-height:1.15;margin:0 0 18px;font-weight:700;min-height:40px}
pre{font:inherit;font-size:18px;white-space:pre-wrap;margin:0}
.empty{color:var(--mute);margin-top:30vh;text-align:center}
.flash{animation:f .6s ease}@keyframes f{from{background:rgba(240,184,79,.35)}to{background:transparent}}
@media (max-width:600px){.app{grid-template-columns:1fr}aside{display:none}main{padding:28px 16px}}
</style></head><body>
<div class="app"><aside><h1>Jev Notes</h1><div id="list"></div></aside>
<main id="main"><div class="empty">No note open yet</div></main></div>
<script>
let last="";
async function tick(){
  try{
    const r=await fetch("/api/state",{cache:"no-store"});const s=await r.json();
    if(s.close){window.close();return}
    const sig=JSON.stringify(s);if(sig===last)return;last=sig;
    document.getElementById("list").innerHTML=s.notes.map(n=>
      `<div class="n ${n.name===s.current?"on":""}"><b>${esc(n.title||"New Note")}</b><span>${n.when}</span></div>`).join("");
    const m=document.getElementById("main");
    if(!s.current){m.innerHTML='<div class="empty">No note open yet</div>';return}
    const lines=s.content.split("\\n");const title=lines[0]||"";const body=lines.slice(1).join("\\n").replace(/^\\n+/,"");
    m.innerHTML=`<div class="date">${s.when}</div><h2 class="flash">${esc(title)}</h2><pre>${esc(body)}</pre>`;
  }catch(e){}
}
function esc(t){return t.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
setInterval(tick,250);tick();
</script></body></html>"""


class NotesApp:
    def __init__(self, notes_dir: Path, browser: str = "Google Chrome", port: int = PORT):
        self.dir = notes_dir
        self.browser = browser
        self.port = port
        self.current: Path | None = None
        self.last_seen = 0.0
        self.close_requested = False
        self._server: ThreadingHTTPServer | None = None

    # -- server --------------------------------------------------------------------------
    def start(self) -> None:
        if self._server:
            return
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/api/state"):
                    app.last_seen = time.monotonic()
                    body = json.dumps(app.state()).encode()
                    if app.close_requested:
                        app.close_requested = False
                    ctype = "application/json"
                elif self.path in ("/", "/index.html"):
                    body, ctype = PAGE.encode(), "text/html; charset=utf-8"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None

    def state(self) -> dict:
        notes = []
        if self.dir.exists():
            for p in sorted(self.dir.glob("*.txt"), key=lambda p: -p.stat().st_mtime):
                try:
                    first = p.read_text(encoding="utf-8").split("\n")[0].strip()
                except OSError:
                    continue
                notes.append({"name": p.name, "title": first, "when": time.strftime("%H:%M", time.localtime(p.stat().st_mtime))})
        content, when = "", ""
        if self.current and self.current.exists():
            content = self.current.read_text(encoding="utf-8")
            when = time.strftime("%-d %B %Y at %H:%M", time.localtime(self.current.stat().st_mtime))
        return {"notes": notes, "current": self.current.name if self.current and self.current.exists() else "",
                "content": content, "when": when, "close": self.close_requested}

    # -- window --------------------------------------------------------------------------
    @property
    def visible(self) -> bool:
        return time.monotonic() - self.last_seen < 1.5

    def show(self) -> None:
        self.start()
        self.close_requested = False
        if self.visible:
            # already on screen: bring Chrome forward
            subprocess.run(["open", "-a", self.browser], capture_output=True)
            return
        subprocess.run(["open", "-na", self.browser, "--args", f"--app=http://127.0.0.1:{self.port}/",
                        "--window-size=900,640"], capture_output=True)

    def hide(self) -> None:
        self.close_requested = True


_APP: NotesApp | None = None


def notes_app(notes_dir: Path, browser: str = "Google Chrome") -> NotesApp:
    """One notes window and one local server per process."""
    global _APP
    if _APP is None:
        _APP = NotesApp(notes_dir, browser)
    else:
        _APP.dir, _APP.browser = notes_dir, browser
    return _APP
