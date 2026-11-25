from uuid import uuid4

import pytest
from pydantic import ValidationError

from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.events import (
    AssistantMessagePayload, HistoryError, SessionCreatedPayload, ToolResultPayload,
    TurnFinishedPayload, UserMessagePayload,
)
from minicodeagent.context.writer import EventWriteError, MemoryEventSink, SessionEventWriter
from minicodeagent.protocol import ToolCall


def test_writer_and_builder_project_only_closed_tool_transactions():
    notices = []
    writer = SessionEventWriter(on_event=notices.append)
    turn_id = uuid4()
    first, second = (
        ToolCall(provider_call_id=provider_id, name="read_file", arguments_raw="{}")
        for provider_id in ("first", "second")
    )
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="inspect"))
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content="", finish_reason="tool_calls",
        tool_calls=(first, second),
    ))
    builder = ContextBuilder()
    with pytest.raises(HistoryError, match="unsettled"):
        builder.project(writer.history, "system")
    with pytest.raises(EventWriteError):
        writer.append("tool_result", ToolResultPayload(
            turn_id=turn_id, call_id=second.call_id, status="ok", output="out of order",
        ))
    for call, output in ((first, "one"), (second, "two")):
        result = ToolResultPayload(
            turn_id=turn_id, call_id=call.call_id, status="ok", output=output,
        )
        writer.append("tool_result", result)
        committed = writer.history
        with pytest.raises(EventWriteError) as exc_info:
            writer.append("tool_result", result)
        assert isinstance(exc_info.value.__cause__, HistoryError)
        assert writer.history == committed
    writer.append("turn_finished", TurnFinishedPayload(turn_id=turn_id, status="completed", reason="test"))
    view = builder.project(writer.history, "system")
    assert [msg.role for msg in view.messages] == ["system", "user", "assistant", "tool", "tool"]
    assert view.messages[2].tool_calls == (first, second)
    assert [msg.call_id for msg in view.messages[3:]] == [first.call_id, second.call_id]
    assert view.consumed_call_ids == (first.call_id, second.call_id)
    assert [event.seq for event in notices] == [1, 2, 3, 4, 5]
    assert len({event.event_id for event in notices}) == 5
    assert len({event.session_id for event in notices}) == 1
    assert all(event.ts.utcoffset().total_seconds() == 0 for event in notices)


def test_failed_sink_does_not_advance_sequence_or_notify():
    class FailingOnce(MemoryEventSink):
        fail = True

        def append(self, event):
            if self.fail:
                raise OSError("temporary failure")
            super().append(event)

    sink = FailingOnce()
    notices = []
    writer = SessionEventWriter(sink=sink, on_event=notices.append)
    payload = UserMessagePayload(turn_id=uuid4(), content="inspect")
    with pytest.raises(EventWriteError):
        writer.append("user_message", payload)
    assert not writer.history and not notices
    sink.fail = False
    assert writer.append("user_message", payload).seq == 1
    assert len(notices) == 1


def test_creation_requires_normalized_absolute_workspace_and_paired_parent(tmp_path):
    workspace = str(tmp_path)
    parent_id = uuid4()
    for invalid in ("relative/path", str(tmp_path / ".." / "other")):
        with pytest.raises(ValidationError):
            SessionCreatedPayload(title="title", workspace=invalid)
    with pytest.raises(ValidationError):
        SessionCreatedPayload(title="title", workspace=workspace, parent_session_id=parent_id)
    payload = SessionCreatedPayload(
        title="title", workspace=workspace, parent_session_id=parent_id, parent_through_seq=3,
    )
    writer = SessionEventWriter()
    event = writer.append("session_created", payload)
    assert event.seq == 1
    with pytest.raises(EventWriteError):
        writer.append("session_created", payload)
    assert writer.history == (event,)
    builder = ContextBuilder()
    assert builder.project((event,), "system") == builder.project((), "system")
    with pytest.raises(HistoryError, match="first"):
        builder.project((event, event), "system")
