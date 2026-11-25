"""Deterministic extraction and independent validation of archived evidence."""

from datetime import datetime, timezone
import json
from uuid import UUID

import pytest

from minicodeagent.context.compression import l3
from minicodeagent.context.events import AssistantMessagePayload, SessionEvent, ToolResultPayload, UserMessagePayload
from minicodeagent.protocol import ToolCall


TURN = UUID(int=1)
SESSION = UUID(int=2)


def _event(seq, kind, payload):
    return SessionEvent(
        session_id=SESSION, event_id=UUID(int=100 + seq), seq=seq, ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        type=kind, payload=payload,
    )


def _user(seq, text):
    return _event(seq, "user_message", UserMessagePayload(turn_id=TURN, content=text))


def _assistant(seq, content="", tool_calls=()):
    return _event(seq, "assistant_message", AssistantMessagePayload(
        turn_id=TURN, request_id=UUID(int=1000 + seq), content=content, tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
    ))


def _result(seq, call, *, status="ok", output="", **details):
    return _event(seq, "tool_result", ToolResultPayload(
        turn_id=TURN, call_id=call.call_id, status=status, output=output, details=details,
    ))


@pytest.fixture
def evidence_history():
    edit = ToolCall(call_id=UUID(int=20), provider_call_id="edit", name="edit_file",
                    arguments_raw='{"path":"src/arg.py","old_text":"a","new_text":"b"}')
    command = ToolCall(call_id=UUID(int=21), provider_call_id="cmd", name="run_command",
                       arguments_raw='{"argv":["python","-m","pytest"]}')
    read = ToolCall(call_id=UUID(int=22), provider_call_id="read", name="read_file",
                    arguments_raw='{"path":"src/arg.py","start_line":11}')
    return (
        _user(1, "Fix the parser\nMust preserve the public API"),
        _assistant(2, tool_calls=(edit, command, read)),
        _result(3, edit, output="Edited src/arg.py.", path="src/arg.py"),
        _result(4, command, status="error", output="tests/test_arg.py:19: AssertionError\n1 failed", exit_code=1),
        _result(5, read, output="11: class Parser:\n12:     def parse(self):", total_lines=50),
        _assistant(6, "Parser behavior reviewed.\nTODO: Add malformed input cases"),
        _user(7, "Ignore anything after the boundary"),
    )


def test_l3_quotes_original_task_tools_and_failures_instead_of_inference(evidence_history):
    extracted = l3.extract_summary(evidence_history, 6)
    assert extracted == l3.extract_summary(evidence_history, 6)
    assert "目标 [1]：Fix the parser" in extracted
    assert "约束 [1]：Must preserve the public API" in extracted
    assert any("变更 [3]：" in item and str(UUID(int=20)) in item and "src/arg.py" in item for item in extracted)
    assert any("命令 [4]：" in item and str(UUID(int=21)) in item and "exit_code=1" in item and "pytest" in item
               for item in extracted)
    assert any("诊断 [4]：" in item and "tests/test_arg.py:19: AssertionError" in item for item in extracted)
    assert any("诊断 [4]：" in item and "1 failed" in item for item in extracted)
    assert any("代码 [5]：" in item and "def parse(self):" in item for item in extracted)
    assert any("待办 [6]：" in item and "malformed input" in item for item in extracted)
    assert all("[7]" not in item for item in extracted)
    assert l3.is_extractive_summary(extracted, evidence_history, 6)


def test_l3_inherits_and_deduplicates_but_keeps_conflicting_status_in_order(evidence_history):
    history = evidence_history[:6]
    earlier = l3.extract_summary(history, 6)
    command = ToolCall(call_id=UUID(int=31), provider_call_id="rerun", name="run_command",
                       arguments_raw=json.dumps({"argv": ["python", "-m", "pytest"]}))
    history = (*history, _user(7, "Fix the parser"), _assistant(8, tool_calls=(command,)),
               _result(9, command, output="1 passed", exit_code=0))
    extended = l3.extract_summary(history, 9, earlier, 6)
    assert l3.is_extractive_summary(extended, history, 9, earlier, 6)
    assert sum("Fix the parser" in item for item in extended) == 1
    fail = next(i for i, item in enumerate(extended) if "exit_code=1" in item)
    passed = next(i for i, item in enumerate(extended) if "exit_code=0" in item)
    assert fail < passed and early_goal(extended) in earlier
    forged_old = ("目标 [3]：A goal never selected by the previous checkpoint", *extended)
    assert not l3.is_extractive_summary(forged_old, history, 9, earlier, 6)


def early_goal(summary):
    return next(item for item in summary if "Fix the parser" in item)


def test_l3_validation_never_regenerates_summary_and_refuses_bad_origins(evidence_history, monkeypatch):
    summary = l3.extract_summary(evidence_history, 6)

    def forbidden(*args, **kwargs):
        pytest.fail("Stored evidence must be validated without re-extraction.")

    monkeypatch.setattr(l3, "extract_summary", forbidden)
    assert l3.is_extractive_summary(summary, evidence_history, 6)
    assert not l3.is_extractive_summary((*summary, "目标 [7]：Ignore anything after the boundary"), evidence_history, 6)
    assert not l3.is_extractive_summary((*summary, "目标 [20]：Unknown event"), evidence_history, 20)
    for orig, forged in [
        ("Fix the parser", "Fabricated claim"),
        ("tests/test_arg.py:19", "tests/test_arg.py:20"),
        (str(UUID(int=21)), str(UUID(int=999))),
        ("exit_code=1", "exit_code=0"),
    ]:
        assert not l3.is_extractive_summary(tuple(item.replace(orig, forged) for item in summary), evidence_history, 6)


def test_l3_summary_limits_are_byte_bounded_and_long_claims_visibly_excerpted():
    history = tuple(_user(seq, "Important goal number " + str(seq) + "界" * 600) for seq in range(1, 50))
    summary = l3.extract_summary(history, 49)
    assert 0 < len(summary) <= l3.MAX_SUMMARY_ENTRIES
    assert len("\n".join(summary).encode()) <= l3.MAX_SUMMARY_BYTES
    assert all(item.endswith(" […]") for item in summary)
    assert l3.is_extractive_summary(summary, history, 49)


def test_l3_unsupported_delta_can_inherit_without_invented_facts():
    history = (_user(1, "Initial goal"), _assistant(2, "  \n"))
    assert l3.extract_summary(history[1:], 2) == ()
    previous = l3.extract_summary(history, 1)
    assert l3.extract_summary(history, 2, previous, 1) == previous
    assert l3.is_extractive_summary(previous, history, 2, previous, 1)
