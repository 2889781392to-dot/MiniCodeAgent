"""Literal evidence extraction for L3 checkpoints over validated history."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import re
from typing import TYPE_CHECKING
from uuid import UUID

from ...protocol import ToolCall, ToolResult, parse_tool_arguments

if TYPE_CHECKING:
    from ..events import SessionEvent


MAX_SUMMARY_ENTRIES = 24
MAX_SUMMARY_BYTES = 6144
_MAX_LITERAL_BYTES = 512
_EXCERPT_MARK = " […]"
_ENTRY = re.compile(r"([^\s]+) \[([1-9][0-9]*)\]：(.+)")
_CONSTRAINT = re.compile(r"\b(?:must|only|do not|never|without|preserve|keep)\b|必须|不得|不要|禁止|约束|保持", re.I)
_TODO = re.compile(
    r"^\s*next\s*:|\b(?:todo|next steps?|not yet|outstanding|remaining|pending)\b|待办|下一步|尚未|未完成",
    re.I,
)
_DECISION = re.compile(r"\b(?:decided|decision|chose|chosen)\b|决定|取舍|改用", re.I)
_ERROR = re.compile(
    r"\b(?:[a-z_]*(?:error|exception)s?|fail(?:ed|ure|ures)?|fatal|traceback|warn(?:ings?)?|panic)\b"
    r"|^\s*(?:E\s|>\s)|错误|失败|异常",
    re.I,
)
_STAT = re.compile(r"\b(?:\d+\s+(?:passed|skipped|failed|errors?|warnings?|xfailed|xpassed)|collected\s+\d+)\b", re.I)
_LOCATION = re.compile(r":\d+\b|\bline\s+\d+\b|[^\s]+\.(?:pyi?|rs|js|ts|go|java|cpp|toml|json|ya?ml)\b", re.I)
_DEFINITION = re.compile(r"^\s*\d+:\s+(?:async\s+def|def|class|function|fn|func)\s+\w+")
_MATCH = re.compile(r".+?:[1-9]\d*: ")
_PRIORITY = {
    "目标": 0, "约束": 0, "待办": 0, "决策": 0, "变更": 0, "命令": 0, "诊断": 1,
    "助手报告": 2, "代码": 2, "匹配": 2, "读取": 3, "搜索": 3, "工具": 3,
}
_TOOL_LABELS = {"变更", "命令", "诊断", "代码", "匹配", "读取", "搜索", "工具"}


@dataclass(frozen=True, slots=True)
class _Evidence:
    label: str
    seq: int
    text: str
    call_id: UUID | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.label, self.text

    def render(self) -> str:
        call = f"{self.call_id} " if self.call_id is not None else ""
        return f"{self.label} [{self.seq}]：{call}{self.text}"


def _fragment(line: str) -> str:
    encoded = line.encode("utf-8")
    if len(encoded) <= _MAX_LITERAL_BYTES:
        return line
    limit = _MAX_LITERAL_BYTES - len(_EXCERPT_MARK.encode("utf-8"))
    return encoded[:limit].decode("utf-8", errors="ignore") + _EXCERPT_MARK


def _parse_entry(value: str) -> _Evidence | None:
    if not isinstance(value, str) or len(value) > _MAX_LITERAL_BYTES + 80:
        return None
    match = _ENTRY.fullmatch(value)
    if match is None:
        return None
    label, seq, content = match.groups()
    if label not in _PRIORITY:
        return None
    call_id = None
    if label in _TOOL_LABELS:
        raw_id, sep, content = content.partition(" ")
        if not sep:
            return None
        try:
            call_id = UUID(raw_id)
        except ValueError:
            return None
        if str(call_id) != raw_id:
            return None
    return _Evidence(label, int(seq), content, call_id) if content.strip() else None


def _sources(
    events: Sequence[SessionEvent], start: int, end: int,
) -> tuple[dict[int, SessionEvent], dict[UUID, ToolCall]]:
    selected = {event.seq: event for event in events if start < event.seq <= end}
    calls = {
        call.call_id: call
        for event in selected.values() if event.type == "assistant_message"
        for call in event.payload.tool_calls
    }
    return selected, calls


def _dialogue_lines(event: SessionEvent) -> Iterable[tuple[str, str]]:
    for pos, line in enumerate(line for line in event.payload.content.splitlines() if line.strip()):
        if pos == 0:
            yield ("目标" if event.type == "user_message" else "助手报告"), line
        elif _TODO.search(line):
            yield "待办", line
        elif _CONSTRAINT.search(line) and event.type == "user_message":
            yield "约束", line
        elif _DECISION.search(line) and event.type == "assistant_message":
            yield "决策", line


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _tool_fact(call: ToolCall, result: ToolResult) -> tuple[str, str]:
    status = f"status={result.status}"
    if call.name == "run_command":
        code = result.details.get("exit_code")
        text = f"{status} exit_code={code}" if type(code) is int else status
        try:
            argv = parse_tool_arguments(call.arguments_raw).get("argv")
        except (ValueError, RecursionError, OverflowError):
            argv = None
        if isinstance(argv, list) and argv and all(isinstance(arg, str) for arg in argv):
            text += f" argv={_json(argv)}"
        return "命令", _fragment(text)
    if call.name in {"write_file", "edit_file"} and result.status == "ok":
        path = result.details.get("path")
        if isinstance(path, str) and path.strip():
            return "变更", _fragment(f"{call.name} {status} path={_json(path)}")
    if call.name in {"read_file", "search_code"}:
        try:
            arguments = parse_tool_arguments(call.arguments_raw)
        except (ValueError, RecursionError, OverflowError):
            arguments = {}
        path = arguments.get("path")
        field = "total_lines" if call.name == "read_file" else "count"
        amount = result.details.get(field)
        text = status
        if isinstance(path, str) and path.strip():
            text += f" requested_path={_json(path)}"
        if type(amount) is int:
            text += f" {field}={amount}"
        return ("读取" if call.name == "read_file" else "搜索"), _fragment(text)
    return "工具", _fragment(f"{call.name} {status}")


def _tool_lines(call: ToolCall, result: ToolResult) -> Iterable[tuple[str, str]]:
    yield _tool_fact(call, result)
    if call.name == "run_command":
        lines = [line for line in result.output.splitlines() if line.strip()]
        # Keep representatives of distinct diagnostic categories from the original output.
        for pattern in (_ERROR, _LOCATION, _STAT):
            source = reversed(lines) if pattern is _STAT else lines
            line = next((line for line in source if pattern.search(line)), None)
            if line is not None:
                yield "诊断", _fragment(line)
    elif call.name in {"read_file", "search_code"}:
        pattern = _DEFINITION if call.name == "read_file" else _MATCH
        label = "代码" if call.name == "read_file" else "匹配"
        matches = (line for line in result.output.splitlines() if pattern.match(line))
        for _, line in zip(range(2), matches):
            yield label, _fragment(line)


def extract_summary(
    events: Sequence[SessionEvent], covered_through_seq: int,
    previous_summary: tuple[str, ...] = (), previous_covered_through_seq: int = 0,
) -> tuple[str, ...]:
    """Merge bounded original evidence with retained items from the prior checkpoint."""
    unique: dict[tuple[str, str], _Evidence] = {}
    for item in previous_summary:
        entry = _parse_entry(item)
        if entry is None:
            raise ValueError("Invalid inherited evidence.")
        unique.setdefault(entry.key, entry)
    selected, calls = _sources(events, previous_covered_through_seq, covered_through_seq)
    for event in selected.values():
        if event.type in ("user_message", "assistant_message"):
            entries = (_Evidence(label, event.seq, _fragment(line)) for label, line in _dialogue_lines(event))
        elif event.type == "tool_result" and (call := calls.get(event.payload.call_id)) is not None:
            entries = (_Evidence(label, event.seq, line, call.call_id)
                       for label, line in _tool_lines(call, event.payload))
        else:
            continue
        for entry in entries:
            unique.setdefault(entry.key, entry)

    prioritized = sorted(unique.values(), key=lambda entry: (_PRIORITY[entry.label], -entry.seq))
    picked: list[_Evidence] = []
    used_bytes = 0
    for entry in prioritized:
        if len(picked) == MAX_SUMMARY_ENTRIES:
            break
        size = len(entry.render().encode("utf-8")) + bool(picked)
        if used_bytes + size <= MAX_SUMMARY_BYTES:
            picked.append(entry)
            used_bytes += size
    return tuple(entry.render() for entry in sorted(picked, key=lambda entry: entry.seq))


def _is_supported(entry: _Evidence, event: SessionEvent, calls: dict[UUID, ToolCall]) -> bool:
    if entry.call_id is None:
        return event.type in ("user_message", "assistant_message") and any(
            label == entry.label and _fragment(line) == entry.text for label, line in _dialogue_lines(event)
        )
    if event.type != "tool_result" or event.payload.call_id != entry.call_id:
        return False
    call = calls.get(entry.call_id)
    if call is None:
        return False
    if entry.label in {"诊断", "代码", "匹配"}:
        pattern = None
        if entry.label == "诊断" and call.name == "run_command":
            return any(
                (_ERROR.search(line) or _LOCATION.search(line) or _STAT.search(line)) and _fragment(line) == entry.text
                for line in event.payload.output.splitlines()
            )
        if entry.label == "代码" and call.name == "read_file":
            pattern = _DEFINITION
        if entry.label == "匹配" and call.name == "search_code":
            pattern = _MATCH
        return pattern is not None and any(
            pattern.match(line) and _fragment(line) == entry.text for line in event.payload.output.splitlines()
        )
    return _tool_fact(call, event.payload) == (entry.label, entry.text)


def is_extractive_summary(
    summary: tuple[str, ...], events: Sequence[SessionEvent], covered_through_seq: int,
    previous_summary: tuple[str, ...] = (), previous_covered_through_seq: int = 0,
) -> bool:
    """Check bounded, ordered source citations without applying the selection strategy."""
    if not summary or len(summary) > MAX_SUMMARY_ENTRIES:
        return False
    try:
        if len("\n".join(summary).encode("utf-8")) > MAX_SUMMARY_BYTES:
            return False
        inherited = set(previous_summary)
        selected, calls = _sources(events, previous_covered_through_seq, covered_through_seq)
        seen: set[tuple[str, str]] = set()
        last_seq = 0
        for item in summary:
            entry = _parse_entry(item)
            if entry is None or not last_seq <= entry.seq <= covered_through_seq or entry.key in seen:
                return False
            if entry.seq <= previous_covered_through_seq:
                if item not in inherited:
                    return False
            elif (event := selected.get(entry.seq)) is None or not _is_supported(entry, event, calls):
                return False
            seen.add(entry.key)
            last_seq = entry.seq
    except (TypeError, ValueError, UnicodeError):
        return False
    return True
