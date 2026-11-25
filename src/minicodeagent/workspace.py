"""Resolve tool paths against a real workspace root."""

import os
import stat
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Literal

EntryKind = Literal["file", "dir", "file_or_dir", "write"]
IGNORED_NAMES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", "node_modules",
})
MAX_SCAN_ENTRIES = 2_000


class WorkspaceError(ValueError):
    pass


class Workspace:
    def __init__(self, root: Path) -> None:
        try:
            self.root = root.resolve(strict=True)
            if not self.root.is_dir():
                raise WorkspaceError("Workspace must be a directory.")
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, WorkspaceError):
                raise
            raise WorkspaceError("Workspace must be an existing directory.") from None

    def resolve_path(self, path: str, *, expect: EntryKind) -> Path:
        try:
            raw = Path(path)
            if raw.anchor and not raw.is_absolute():
                raise WorkspaceError("Drive-relative or rooted paths must be fully qualified.")
            if raw.is_absolute() and raw.anchor.casefold() != self.root.anchor.casefold():
                raise WorkspaceError("Path is outside the workspace.")
            if os.name == "nt" and (
                raw.is_reserved() or any(":" in part for part in raw.parts if part != raw.anchor)
            ):
                raise WorkspaceError("Device paths and alternate data streams are unsupported.")
            target = (self.root / raw).resolve()
            if not target.is_relative_to(self.root):
                raise WorkspaceError("Path is outside the workspace.")
            if expect == "write":
                if target.exists() and not target.is_file():
                    raise WorkspaceError("Write target must be a regular file.")
            elif expect == "file":
                if not target.is_file():
                    raise WorkspaceError("Path must be an existing regular file.")
            elif expect == "dir":
                if not target.is_dir():
                    raise WorkspaceError("Path must be an existing directory.")
            elif not (target.is_file() or target.is_dir()):
                raise WorkspaceError("Path must be an existing file or directory.")
            return target
        except WorkspaceError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise WorkspaceError("Cannot resolve the requested workspace path.") from None

    def relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except (OSError, RuntimeError, ValueError):
            raise WorkspaceError("Path is outside the workspace or cannot be resolved.") from None

    def walk_files(self, start: Path) -> Iterator[Path]:
        """Yield regular files in relative path order, with a bounded directory scan."""
        if self._is_ignored(start):
            return
        if start.is_file():
            yield start
            return
        remaining = MAX_SCAN_ENTRIES
        pending = [start]
        while pending:
            path = pending.pop()
            if path.is_file():
                yield path
                continue
            if remaining == 0:
                return
            try:
                path = self.resolve_path(str(path), expect="dir")
                with os.scandir(path) as entries:
                    children = []
                    for entry in islice(entries, remaining):
                        remaining -= 1
                        child = Path(entry.path)
                        if self._is_ignored(child):
                            continue
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                            if stat.S_ISLNK(metadata.st_mode) or (
                                getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                            ):
                                continue
                            if stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode):
                                children.append(self.resolve_path(str(child), expect="file_or_dir"))
                        except (OSError, WorkspaceError):
                            continue
                pending.extend(sorted(
                    children,
                    key=lambda child: child.name + ("/" if child.is_dir() else ""),
                    reverse=True,
                ))
            except (OSError, WorkspaceError):
                continue

    def _is_ignored(self, path: Path) -> bool:
        return any(part.casefold() in IGNORED_NAMES for part in path.relative_to(self.root).parts)
