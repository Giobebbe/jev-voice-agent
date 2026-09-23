"""The sandbox is the hard constraint: nothing is created or deleted outside the root."""

from pathlib import Path

import pytest

from jevagent import config
from jevagent.sandbox import Sandbox, SandboxError, clean_name


@pytest.fixture
def sb(tmp_path, monkeypatch):
    root = tmp_path / "Folder JEV"
    root.mkdir()
    monkeypatch.setattr("jevagent.sandbox.ALLOWED_ROOTS", (root,))
    return Sandbox(root)


def test_rejects_root_not_allowlisted(tmp_path):
    with pytest.raises(SandboxError):
        Sandbox(tmp_path)


def test_real_allowlist_is_only_the_two_jev_folders():
    names = {p.name for p in config.ALLOWED_ROOTS}
    assert names == {"Folder JEV", "Folder JEV copy"}
    assert all(p.parent == Path.home() / "Desktop" for p in config.ALLOWED_ROOTS)


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x", "a/../../x", "..", "Notes/../../y"])
def test_path_traversal_rejected(sb, bad):
    with pytest.raises(SandboxError):
        sb.path(bad)


def test_root_itself_rejected_for_delete(sb):
    with pytest.raises(SandboxError):
        sb.delete("")
    with pytest.raises(SandboxError):
        sb.delete(".")


def test_symlink_escape_rejected(sb, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    (sb.root / "link").symlink_to(outside)
    with pytest.raises(SandboxError):
        sb.delete("link/secret.txt")
    with pytest.raises(SandboxError):
        sb.path("link")
    assert (outside / "secret.txt").exists()
    assert "link/" not in sb.listing()


@pytest.mark.parametrize("name,expected", [
    ("../../etc", "etc"), ("a/b", "a b"), ("  hello  ", "hello"), ("x:y", "x y"), ("..hidden", "hidden"),
])
def test_clean_name(name, expected):
    assert clean_name(name) == expected


@pytest.mark.parametrize("name", ["", "   ", "..", "/"])
def test_clean_name_rejects_empty(name):
    with pytest.raises(SandboxError):
        clean_name(name)


def test_create_rename_move_delete_stay_inside(sb):
    f = sb.create_folder("Projects")
    note = sb.create_file("hello", "hi", parent_rel="")
    assert note.name == "hello.txt" and note.parent == sb.root
    moved = sb.move("hello.txt", "Projects/")
    assert moved == f / "hello.txt"
    renamed = sb.rename("Projects/hello.txt", "../../evil")
    assert renamed.parent == f and renamed.name == "evil.txt"
    sb.delete("Projects/")
    assert list(sb.root.iterdir()) == []


def test_unique_names_do_not_overwrite(sb):
    a = sb.create_file("same", "1")
    b = sb.create_file("same", "2")
    assert a != b and a.read_text() == "1" and b.read_text() == "2"


def test_move_folder_into_itself_rejected(sb):
    sb.create_folder("A")
    sb.create_folder("B", parent_rel="A")
    with pytest.raises(SandboxError):
        sb.move("A/", "A/B/")


def test_listing_hides_dotfiles(sb):
    (sb.root / ".DS_Store").write_text("")
    sb.create_folder("Photos")
    assert sb.listing() == ["Photos/"]
