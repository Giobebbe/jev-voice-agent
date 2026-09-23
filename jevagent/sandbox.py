"""Every filesystem write the agent makes goes through Sandbox.

The root must be one of config.ALLOWED_ROOTS. Paths are resolved and must stay strictly
inside the root; absolute paths, '..', and symlinks pointing outside are rejected.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import ALLOWED_ROOTS


class SandboxError(Exception):
    pass


_BAD_CHARS = re.compile(r"[/\\:\x00-\x1f]")


def clean_name(name: str, max_len: int = 80) -> str:
    """Turn a spoken name into a safe single path component."""
    name = _BAD_CHARS.sub(" ", name or "")
    words = [w for w in name.split() if w.strip(".")]  # drop '.', '..' fragments
    name = " ".join(words).strip(". ")
    if not name or name in {".", ".."}:
        raise SandboxError("empty or invalid name")
    return name[:max_len]


class Sandbox:
    def __init__(self, root: Path | str):
        root = Path(root).expanduser()
        allowed = {r.resolve() for r in ALLOWED_ROOTS}
        if root.is_symlink():
            raise SandboxError(f"root must not be a symlink: {root}")
        resolved = root.resolve()
        if resolved not in allowed:
            raise SandboxError(f"root {resolved} is not an allowed root {sorted(map(str, allowed))}")
        if not resolved.is_dir():
            raise SandboxError(f"root does not exist: {resolved}")
        self.root = resolved

    # -- path guard -------------------------------------------------------------------
    def path(self, rel: str | Path, *, allow_root: bool = False) -> Path:
        rel = Path(rel)
        if rel.is_absolute():
            raise SandboxError(f"absolute path rejected: {rel}")
        if any(part == ".." for part in rel.parts):
            raise SandboxError(f"parent traversal rejected: {rel}")
        target = self.root / rel
        # Every existing component must be a real directory/file inside the root.
        probe = self.root
        for part in rel.parts:
            probe = probe / part
            if probe.is_symlink():
                raise SandboxError(f"symlink rejected: {probe}")
        resolved = target.resolve()
        if resolved == self.root:
            if allow_root:
                return resolved
            raise SandboxError("operation on the root itself rejected")
        if self.root not in resolved.parents:
            raise SandboxError(f"escapes sandbox: {resolved}")
        return resolved

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    # -- queries ----------------------------------------------------------------------
    def listing(self, max_items: int = 120, depth: int = 2) -> list[str]:
        items: list[str] = []

        def walk(d: Path, level: int) -> None:
            for child in sorted(d.iterdir(), key=lambda c: c.name.lower()):
                if child.name.startswith(".") or child.is_symlink():
                    continue
                items.append(self.rel(child) + ("/" if child.is_dir() else ""))
                if len(items) >= max_items:
                    return
                if child.is_dir() and level < depth:
                    walk(child, level + 1)

        walk(self.root, 1)
        return items[:max_items]

    def folders(self) -> list[str]:
        return [i for i in self.listing() if i.endswith("/")]

    def unique(self, parent: Path, name: str) -> Path:
        candidate = parent / name
        if not candidate.exists():
            return candidate
        stem, suffix = Path(name).stem, Path(name).suffix
        n = 2
        while (parent / f"{stem} {n}{suffix}").exists():
            n += 1
        return parent / f"{stem} {n}{suffix}"

    # -- writes -----------------------------------------------------------------------
    def ensure_dir(self, rel: str) -> Path:
        d = self.path(rel)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_folder(self, name: str, parent_rel: str = "") -> Path:
        parent = self.path(parent_rel, allow_root=True) if parent_rel else self.root
        target = self.unique(parent, clean_name(name))
        self.path(self.rel(target))  # re-check
        target.mkdir()
        return target

    def create_file(self, name: str, text: str = "", parent_rel: str = "", default_suffix: str = ".txt") -> Path:
        name = clean_name(name)
        if not Path(name).suffix:
            name += default_suffix
        parent = self.path(parent_rel, allow_root=True) if parent_rel else self.root
        target = self.unique(parent, name)
        self.path(self.rel(target))
        target.write_text(text, encoding="utf-8")
        return target

    def write_file(self, rel: str, text: str) -> Path:
        target = self.path(rel)
        target.write_text(text, encoding="utf-8")
        return target

    def delete(self, rel: str) -> Path:
        target = self.path(rel.rstrip("/"))
        if not target.exists():
            raise SandboxError(f"not found: {rel}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return target

    def rename(self, rel: str, new_name: str) -> Path:
        src = self.path(rel.rstrip("/"))
        if not src.exists():
            raise SandboxError(f"not found: {rel}")
        new_name = clean_name(new_name)
        if src.is_file() and not Path(new_name).suffix:
            new_name += src.suffix
        dst = self.unique(src.parent, new_name)
        self.path(self.rel(dst))
        src.rename(dst)
        return dst

    def move(self, rel: str, dest_rel: str) -> Path:
        src = self.path(rel.rstrip("/"))
        dest_dir = self.path(dest_rel.rstrip("/"), allow_root=True) if dest_rel not in ("", "/") else self.root
        if not dest_dir.is_dir():
            raise SandboxError(f"destination is not a folder: {dest_rel}")
        if src == dest_dir or src in dest_dir.parents:
            raise SandboxError("cannot move a folder into itself")
        dst = self.unique(dest_dir, src.name)
        self.path(self.rel(dst))
        shutil.move(str(src), str(dst))
        return dst
