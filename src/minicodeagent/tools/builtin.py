"""The six V0.1 workspace tools."""

import os
from itertools import islice
from typing import Annotated
from uuid import UUID

from pydantic import Field

from ..protocol import NonBlank, ProtocolModel, ToolResult
from ..workspace import MAX_SCAN_ENTRIES, Workspace, WorkspaceError
from .process import CommandConfirmation, execute_command
from .registry import ToolDefinition, ToolFailure, ToolRegistry

MAX_FILE_BYTES = 256 * 1024
MAX_SEARCH_BYTES = 4 * 1024 * 1024
MAX_LINE_CHARS = 500
PathArgument = Annotated[NonBlank, Field(max_length=4096)]
ResultLimit = Annotated[int, Field(ge=1, le=200)]


class ListFilesArgs(ProtocolModel):
    path: PathArgument = "."
    limit: ResultLimit = 100


class ReadFileArgs(ProtocolModel):
    path: PathArgument
    start_line: int = Field(default=1, ge=1)
    line_count: int = Field(default=80, ge=1, le=200)


class SearchCodeArgs(ProtocolModel):
    query: str = Field(min_length=1, max_length=200)
    path: PathArgument = "."
    limit: ResultLimit = 50


class WriteFileArgs(ProtocolModel):
    path: PathArgument
    content: str = Field(max_length=MAX_FILE_BYTES)


class EditFileArgs(ProtocolModel):
    path: PathArgument
    old_text: str = Field(min_length=1, max_length=MAX_FILE_BYTES)
    new_text: str = Field(max_length=MAX_FILE_BYTES)


class RunCommandArgs(ProtocolModel):
    argv: list[Annotated[str, Field(max_length=8192)]] = Field(min_length=1, max_length=64)
    cwd: PathArgument = "."
    timeout_seconds: float = Field(default=30.0, gt=0, le=300, allow_inf_nan=False)


def _read_text(workspace: Workspace, path: str) -> str:
    target = workspace.resolve_path(path, expect="file")
    try:
        with target.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ToolFailure(f"File exceeds the {MAX_FILE_BYTES}-byte read limit.")
        if b"\0" in data:
            raise ToolFailure("File appears to be binary.")
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolFailure("File is not UTF-8 text.") from None
    except OSError:
        raise ToolFailure("Cannot read the requested file.") from None


def _write_text(workspace: Workspace, path: str, text: str) -> int:
    data = text.encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise ToolFailure(f"New file content exceeds {MAX_FILE_BYTES} UTF-8 bytes.")
    target = workspace.resolve_path(path, expect="write")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target = workspace.resolve_path(path, expect="write")
        return target.write_bytes(data)
    except OSError:
        raise ToolFailure("Cannot create or write the requested file.") from None


def list_files(args: ListFilesArgs, workspace: Workspace, call_id: UUID) -> ToolResult:
    target = workspace.resolve_path(args.path, expect="dir")
    paths = [
        workspace.relative_path(path)
        for path in islice(workspace.walk_files(target), args.limit + 1)
    ]
    shown = paths[:args.limit]
    return ToolResult(
        call_id=call_id, status="ok",
        output="\n".join(shown) or "No files in the scanned directory.",
        details={"count": len(shown), "more_results": len(paths) > args.limit, "scan_entry_limit": MAX_SCAN_ENTRIES},
    )


def read_file(args: ReadFileArgs, workspace: Workspace, call_id: UUID) -> ToolResult:
    text = _read_text(workspace, args.path)
    lines = text.splitlines()
    end = args.start_line - 1 + args.line_count
    next_line = end + 1 if end < len(lines) else None
    page = "\n".join(
        f"{line_no}: {line[:MAX_LINE_CHARS]}" + (" […]" if len(line) > MAX_LINE_CHARS else "")
        for line_no, line in enumerate(lines[args.start_line - 1:end], args.start_line)
    ) or "No lines in this range."
    if next_line is not None:
        page += f"\n[Next page starts at line {next_line}.]"
    if "\r\n" in text:
        page += "\n[File contains CRLF line endings. Match them literally when editing.]"
    return ToolResult(
        call_id=call_id, status="ok", output=page,
        details={"next_start_line": next_line, "total_lines": len(lines)},
    )


def search_code(args: SearchCodeArgs, workspace: Workspace, call_id: UUID) -> ToolResult:
    target = workspace.resolve_path(args.path, expect="file_or_dir")
    remaining = MAX_SEARCH_BYTES
    matches = []
    for path in workspace.walk_files(target):
        if remaining == 0 or len(matches) == args.limit:
            break
        try:
            relative = workspace.relative_path(path)
            with workspace.resolve_path(relative, expect="file").open("rb") as stream:
                data = stream.read(min(MAX_FILE_BYTES + 1, remaining))
                remaining -= len(data)
                if remaining == 0 and os.fstat(stream.fileno()).st_size != len(data):
                    continue
            if len(data) > MAX_FILE_BYTES or b"\0" in data:
                continue
            lines = data.decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError, WorkspaceError):
            continue
        for line_no, line in enumerate(lines, 1):
            index = line.find(args.query)
            if index == -1:
                continue
            start = max(0, index - 80)
            snippet = line[start:start + MAX_LINE_CHARS]
            matches.append(f"{relative}:{line_no}: {snippet}")
            if len(matches) == args.limit:
                break
    return ToolResult(
        call_id=call_id, status="ok",
        output="\n".join(matches) or "No matches in the scanned files.",
        details={
            "count": len(matches), "result_limit_reached": len(matches) == args.limit,
            "scan_byte_limit_reached": remaining == 0, "scan_entry_limit": MAX_SCAN_ENTRIES,
        },
    )


def write_file(args: WriteFileArgs, workspace: Workspace, call_id: UUID) -> ToolResult:
    target = workspace.resolve_path(args.path, expect="write")
    path = workspace.relative_path(target)
    size = _write_text(workspace, path, args.content)
    return ToolResult(
        call_id=call_id, status="ok", output=f"Wrote {path}.",
        details={"path": path, "bytes_written": size},
    )


def edit_file(args: EditFileArgs, workspace: Workspace, call_id: UUID) -> ToolResult:
    target = workspace.resolve_path(args.path, expect="file")
    path = workspace.relative_path(target)
    text = _read_text(workspace, path)
    first = text.find(args.old_text)
    if first == -1:
        raise ToolFailure("old_text was not found. Read the file again to get its exact text.")
    if text.find(args.old_text, first + 1) != -1:
        raise ToolFailure("old_text matches more than once. Include more surrounding text.")
    _write_text(workspace, path, text[:first] + args.new_text + text[first + len(args.old_text):])
    return ToolResult(call_id=call_id, status="ok", output=f"Edited {path}.", details={"path": path})


def register_builtin_tools(registry: ToolRegistry, *, confirm_command: CommandConfirmation) -> None:
    def run_command(args: RunCommandArgs, workspace: Workspace, call_id: UUID) -> ToolResult:
        if not args.argv[0].strip() or any("\0" in value for value in args.argv):
            raise ToolFailure("argv must name an executable and contain no null characters.")
        cwd = workspace.resolve_path(args.cwd, expect="dir")
        return execute_command(tuple(args.argv), cwd, args.timeout_seconds, call_id, confirm_command)

    for definition in (
        ToolDefinition("list_files", "List files relative to the workspace, skipping caches.", ListFilesArgs, list_files),
        ToolDefinition("read_file", "Read a page of UTF-8 text with 1-based line numbers.", ReadFileArgs, read_file),
        ToolDefinition("search_code", "Find a literal, case-sensitive string in UTF-8 project files.", SearchCodeArgs, search_code),
        ToolDefinition("write_file", "Create or replace a UTF-8 file, creating its parent directories.", WriteFileArgs, write_file),
        ToolDefinition("edit_file", "Replace old_text at its single exact occurrence in a file.", EditFileArgs, edit_file),
        ToolDefinition("run_command", "Run argv in a workspace directory after user confirmation.", RunCommandArgs, run_command),
    ):
        registry.register(definition)
