"""Read the single project background file from a validated Workspace."""

from .workspace import Workspace, WorkspaceError

MAX_PROJECT_MEMORY_BYTES = 16 * 1024
_MEMORY_FILENAME = "AGENTS.md"


class ProjectMemoryError(ValueError):
    pass


def load_project_memory(workspace: Workspace) -> str | None:
    source = workspace.root / _MEMORY_FILENAME
    try:
        source.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProjectMemoryError(f"Could not inspect project memory at {source}.") from exc

    try:
        path = workspace.resolve_path(_MEMORY_FILENAME, expect="file")
    except WorkspaceError as exc:
        raise ProjectMemoryError(f"Invalid project memory path {source}. {exc}") from exc

    data = bytearray()
    try:
        with path.open("rb", buffering=0) as file:
            while len(data) <= MAX_PROJECT_MEMORY_BYTES:
                chunk = file.read(MAX_PROJECT_MEMORY_BYTES + 1 - len(data))
                if not chunk:
                    break
                data.extend(chunk)
    except OSError as exc:
        raise ProjectMemoryError(f"Could not read project memory at {source}.") from exc
    if len(data) > MAX_PROJECT_MEMORY_BYTES:
        raise ProjectMemoryError(f"Project memory at {source} exceeds the 16 KiB limit.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ProjectMemoryError(f"Project memory at {source} is not valid UTF-8.") from exc
