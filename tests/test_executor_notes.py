"""Notes, rename/move following and trash through the real Executor (no windows opened)."""

import pytest

from jevagent.brain import Arg, Decision
from jevagent.config import Settings
from jevagent.executor import Executor
from jevagent.sandbox import Sandbox


def dec(tool, **args):
    names = ("app", "site", "volume", "mode", "item", "dest", "title", "note_text", "query", "name")
    return Decision(clause="", tool=tool, tool_p=1.0,
                    args={n: Arg(args.get(n), 1.0, omitted=n not in args) for n in names})


@pytest.fixture
def ex(tmp_path, monkeypatch):
    root = tmp_path / "Folder JEV"
    root.mkdir()
    monkeypatch.setattr("jevagent.sandbox.ALLOWED_ROOTS", (root,))
    e = Executor(Settings(root=root), Sandbox(root))
    monkeypatch.setattr(e.notes, "show", lambda: None)
    return e


def test_note_lifecycle(ex):
    assert ex.run(dec("create_note")).ok
    assert ex.desk.current_note == "Notes/Untitled.txt"
    assert ex.run(dec("set_note_title", title="Call Dr. Smith")).ok
    assert ex.desk.current_note == "Notes/Call Dr. Smith.txt"
    assert ex.run(dec("write_in_note", note_text="buy milk")).ok
    text = (ex.sb.root / ex.desk.current_note).read_text()
    assert text == "Call Dr. Smith\n\nbuy milk\n"
    assert ex.notes.state()["current"] == "Call Dr. Smith.txt"


def test_bad_title_leaves_note_untouched(ex):
    ex.run(dec("create_note", title="keep"))
    before = (ex.sb.root / ex.desk.current_note).read_text()
    assert not ex.run(dec("set_note_title", title="...")).ok
    assert (ex.sb.root / "Notes/keep.txt").read_text() == before


def test_current_note_follows_folder_move(ex):
    ex.run(dec("create_note", title="plan"))
    ex.run(dec("create_folder", name="Archive"))
    assert ex.run(dec("move_item", item="Notes/", dest="Archive/")).ok
    assert ex.desk.current_note == "Archive/Notes/plan.txt"
    assert ex.run(dec("write_in_note", note_text="still here")).ok
    assert "still here" in (ex.sb.root / "Archive/Notes/plan.txt").read_text()


def test_delete_goes_to_trash(ex):
    ex.run(dec("create_folder", name="Project"))
    res = ex.run(dec("delete_item", item="Project/"))
    assert res.ok and "trash" in res.say
    assert ex.sb.listing() == [] and (ex.sb.root / ".trash" / "Project").is_dir()
