"""Extract traceable evidence from known command and search tool results."""

from dataclasses import dataclass
import json
import re

from ...protocol import ToolCall, ToolResult, parse_tool_arguments


_MARKER = "[L2 tool result excerpt]\n"
_EDGE_LINES = 2
_SEARCH_MATCH = re.compile(r"(.+?):[1-9][0-9]*: ")
_ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")
_COMMAND_EVIDENCE = re.compile(
    r"\b(?:[a-z_]*(?:error|exception)s?|fail(?:ed|ure|ures)?|fatal|traceback|panic|warn(?:ings?)?"
    r"|assert(?:ion)?|passed|skipped|deselected|xfailed|xpassed)\b"
    r"|\bcollected\s+[0-9]+\s+items?\b"
    r"|^\s*(?:E\s|>\s)"
    r"|\bline\s+[0-9]+\b|\b[^\s:]+:[0-9]+\b"
    r"|\b[^\s:]+\.(?:pyi?|tsx?|jsx?|json|toml|ya?ml|md|txt|cfg|ini|rs|go|java|cpp|c|h)\b"
    r"|\[output truncated:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Section:
    lines: list[str]
    required: set[int]


def _edge_indexes(lines: list[str]) -> set[int]:
    return set(range(1, min(len(lines), _EDGE_LINES) + 1)) | set(
        range(max(1, len(lines) - _EDGE_LINES + 1), len(lines) + 1),
    )


def _command_section(output: str) -> _Section:
    lines = output.splitlines()
    indexes = _edge_indexes(lines)
    seen = {lines[index - 1] for index in indexes}
    for index, line in enumerate(lines, 1):
        if line not in seen and _COMMAND_EVIDENCE.search(_ANSI_COLOR.sub("", line)):
            indexes.add(index)
            seen.add(line)
    return _Section(lines, indexes)


def _command_source(result: ToolResult, args: dict) -> tuple[dict, dict[str, _Section]] | None:
    details = result.details
    argv = args.get("argv")
    cwd = args.get("cwd")
    code = details.get("exit_code")
    stdout, stderr = details.get("stdout"), details.get("stderr")
    if (
        not isinstance(argv, list) or not argv
        or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
        or not argv[0].strip() or ("cwd" in args and (not isinstance(cwd, str) or not cwd.strip()))
        or type(code) is not int or result.status != ("ok" if code == 0 else "error")
        or not isinstance(stdout, str) or not isinstance(stderr, str)
        or result.output != f"Exit code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    ):
        return None
    metadata = {"argv": argv, "exit_code": code}
    if cwd is not None:
        metadata["requested_cwd"] = cwd
    for key in ("stdout_truncated", "stderr_truncated", "capture_incomplete"):
        if key in details:
            if type(details[key]) is not bool:
                return None
            metadata[key] = details[key]
    if "error_code" in details:
        if not isinstance(details["error_code"], str):
            return None
        metadata["error_code"] = details["error_code"]
    return metadata, {"stdout": _command_section(stdout), "stderr": _command_section(stderr)}


def _search_source(result: ToolResult, args: dict) -> tuple[dict, dict[str, _Section]] | None:
    details = result.details
    query, path = args.get("query"), args.get("path")
    count = details.get("count")
    lines = result.output.splitlines()
    if (
        result.status != "ok" or not isinstance(query, str) or not query
        or ("path" in args and (not isinstance(path, str) or not path.strip()))
        or type(count) is not int or count <= 0 or len(lines) != count
    ):
        return None
    metadata = {"query": query, "count": count}
    if path is not None:
        metadata["requested_path"] = path
    for key in ("result_limit_reached", "scan_byte_limit_reached"):
        if key in details:
            if type(details[key]) is not bool:
                return None
            metadata[key] = details[key]
    if "scan_entry_limit" in details:
        if type(details["scan_entry_limit"]) is not int or details["scan_entry_limit"] <= 0:
            return None
        metadata["scan_entry_limit"] = details["scan_entry_limit"]

    files: dict[str, int] = {}
    for index, line in enumerate(lines, 1):
        match = _SEARCH_MATCH.match(line)
        if match is None:
            return None
        files.setdefault(match[1], index)
    return metadata, {"matches": _Section(lines, _edge_indexes(lines) | set(files.values()))}


def _source(seq: int, call: ToolCall, result: ToolResult) -> tuple[dict, dict[str, _Section]] | None:
    if (
        type(seq) is not int or seq < 1 or call.call_id != result.call_id
        or result.status not in {"error", "ok"} or call.name not in {"run_command", "search_code"}
        or result.details.get("execution_outcome") is not None
        or result.details.get("capture_incomplete") or result.details.get("output_truncated")
    ):
        return None
    try:
        args = parse_tool_arguments(call.arguments_raw)
    except (ValueError, RecursionError, OverflowError):
        return None
    source = _command_source(result, args) if call.name == "run_command" else _search_source(result, args)
    if source is None:
        return None
    metadata, sections = source
    return {
        "source_seq": seq, "call_id": str(call.call_id), "tool": call.name, "status": result.status, **metadata,
    }, sections


def _json(value: object, *, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=sort_keys)


def extract_tool_result(seq: int, call: ToolCall, result: ToolResult) -> str | None:
    """Propose a literal excerpt with actual omissions for budget evaluation."""
    source = _source(seq, call, result)
    if source is None:
        return None
    metadata, sections = source
    if all(len(section.required) == len(section.lines) for section in sections.values()):
        return None
    body = dict(metadata)
    for name, section in sections.items():
        body[name] = {
            "omitted_lines": len(section.lines) - len(section.required),
            "lines": [[index, section.lines[index - 1]] for index in sorted(section.required)],
        }
    return _MARKER + _json(body)


def _validated_omissions(value: object, source: _Section) -> int | None:
    if not isinstance(value, dict) or value.keys() != {"omitted_lines", "lines"}:
        return None
    lines, omitted = value["lines"], value["omitted_lines"]
    if not isinstance(lines, list) or type(omitted) is not int or omitted != len(source.lines) - len(lines):
        return None
    previous = 0
    indexes = set()
    for pair in lines:
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        index, line = pair
        if type(index) is not int or not previous < index <= len(source.lines) or line != source.lines[index - 1]:
            return None
        previous = index
        indexes.add(index)
    return omitted if source.required.issubset(indexes) else None


def is_tool_result_excerpt(seq: int, call: ToolCall, result: ToolResult, replacement: str) -> bool:
    """Verify a stored excerpt references the exact source facts and retains essential evidence."""
    if not replacement.startswith(_MARKER):
        return False
    source = _source(seq, call, result)
    if source is None:
        return False
    metadata, sections = source
    try:
        text = replacement.removeprefix(_MARKER)
        body = json.loads(text)
        if not isinstance(body, dict) or _json(body) != text:
            return False
        if _json({key: value for key, value in body.items() if key not in sections}, sort_keys=True) != _json(
            metadata, sort_keys=True,
        ):
            return False
        omissions = [_validated_omissions(body.get(name), section) for name, section in sections.items()]
        return None not in omissions and any(omissions)
    except (ValueError, TypeError, RecursionError, OverflowError):
        return False
