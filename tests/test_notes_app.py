"""The notes server answers only localhost requests that carry the per-run key."""

import http.client
import json

import pytest

from jevagent.notes_app import NotesApp
from jevagent.sandbox import Sandbox


@pytest.fixture
def app(tmp_path, monkeypatch):
    root = tmp_path / "Folder JEV"
    (root / "Notes").mkdir(parents=True)
    (root / "Notes" / "hello.txt").write_text("hello\n\nbody\n")
    monkeypatch.setattr("jevagent.sandbox.ALLOWED_ROOTS", (root,))
    a = NotesApp(Sandbox(root), port=18765)
    a.start()
    yield a
    a.stop()


def get(app, path, host=None):
    c = http.client.HTTPConnection("127.0.0.1", app.port, timeout=3)
    c.request("GET", path, headers={"Host": host or f"127.0.0.1:{app.port}"})
    r = c.getresponse()
    return r.status, r.read()


def test_key_required(app):
    assert get(app, "/api/state")[0] == 403
    assert get(app, "/api/state?k=wrong")[0] == 403
    status, body = get(app, f"/api/state?k={app.key}")
    assert status == 200 and json.loads(body)["notes"][0]["title"] == "hello"


def test_foreign_host_rejected(app):
    assert get(app, f"/api/state?k={app.key}", host="evil.example:18765")[0] == 403


def test_rejected_request_has_no_side_effects(app):
    app.close_requested = True
    get(app, "/api/state")
    assert app.close_requested and not app.visible
