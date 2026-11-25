import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from minicodeagent.agent import AgentLoop, LoopLimits
from minicodeagent.context import store as event_store
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.events import (
    AssistantMessagePayload,
    SessionCreatedPayload,
    SessionEvent,
    ToolResultPayload,
    TurnFinishedPayload,
    UserMessagePayload,
)
from minicodeagent.context.store import (
    EventStoreBusyError,
    EventStoreCorruptError,
    EventStoreError,
    JsonlEventSink,
    JsonlEventStore,
)
from minicodeagent.context.writer import EventWriteError, SessionEventWriter
from minicodeagent.protocol import ModelReply, TokenUsage, ToolCall
from minicodeagent.tools.registry import ToolRegistry
from minicodeagent.workspace import Workspace


def sample_turn(writer):
    turn_id = uuid4()
    call = ToolCall(
        provider_call_id="读取", name="read_file", arguments_raw='{"path":"文件.py"}',
    )
    rejected = ToolCall(
        provider_call_id="deferred", name="read_file", arguments_raw='{"path":"other.py"}',
    )
    writer.append("user_message", UserMessagePayload(
        turn_id=turn_id, content='找出错误\n路径：C:\\项目\t"代码"',
    ))
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content="", finish_reason="tool_calls",
        tool_calls=(call,), usage=TokenUsage(input_tokens=8, output_tokens=3, total_tokens=11),
    ))
    writer.append("tool_result", ToolResultPayload(
        turn_id=turn_id, call_id=call.call_id, status="ok", output="第一行\n第二行",
        details={"exit_code": 0, "extra": [True, None, "中文"]},
    ))
    final = writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content="已读取", finish_reason="tool_calls",
        consumed_call_ids=(call.call_id,), rejected_tool_calls=(rejected,),
    ))
    writer.append("turn_finished", TurnFinishedPayload(
        turn_id=turn_id, status="limited", reason="tool_batch_limit", assistant_seq=final.seq,
    ))


def create_payload(workspace):
    return SessionCreatedPayload(title="会话", workspace=str(workspace.resolve()))


def new_writer(store, session_id=None, **kwargs):
    session_id = session_id or uuid4()
    return SessionEventWriter(
        session_id=session_id, sink=JsonlEventSink(store, session_id), **kwargs,
    )


def log_path(root, session_id):
    return root / "sessions" / f"{session_id}.jsonl"


@pytest.fixture
def memory_history(tmp_path):
    memory = SessionEventWriter()
    memory.append("session_created", create_payload(tmp_path))
    sample_turn(memory)
    return memory.history


def test_appends_existing_event_types_as_utf8_round_trip(tmp_path):
    notices = []
    store = JsonlEventStore(tmp_path)
    writer = new_writer(store, on_event=notices.append)
    first = writer.append("session_created", create_payload(tmp_path))
    path = log_path(tmp_path, first.session_id)
    initial = path.read_bytes()
    assert initial.count(b"\n") == 1 and initial.endswith(b"\n")
    sample_turn(writer)

    raw = path.read_bytes()
    assert raw.startswith(initial) and raw.endswith(b"\n")
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == len(notices) == 6
    assert records[0]["payload"] == {"title": "会话", "workspace": str(tmp_path.resolve())}
    assert [record["seq"] for record in records] == list(range(1, 7))
    assert {record["v"] for record in records} == {1}
    assert {record["session_id"] for record in records} == {str(first.session_id)}
    assert len({record["event_id"] for record in records}) == 6
    assert [record["type"] for record in records] == [
        "session_created", "user_message", "assistant_message", "tool_result",
        "assistant_message", "turn_finished",
    ]
    assert records[1]["payload"]["content"] == notices[1].payload.content
    assert records[2]["payload"]["tool_calls"][0]["call_id"] == str(
        notices[2].payload.tool_calls[0].call_id,
    )

    reopened = JsonlEventStore(tmp_path)
    events = tuple(reopened.iter_events(first.session_id))
    assert events == tuple(notices) == new_writer(reopened, first.session_id).history
    assert all(item.ts.utcoffset().total_seconds() == 0 for item in events)


def test_disk_projection_matches_memory_with_or_without_session_header(tmp_path, memory_history):
    session_id = memory_history[0].session_id
    JsonlEventStore(tmp_path).publish(session_id, iter(memory_history))
    reopened = new_writer(JsonlEventStore(tmp_path), session_id)
    builder = ContextBuilder()
    projection = builder.project(reopened.history, "system")
    assert projection == builder.project(memory_history, "system")
    assert projection == builder.project(memory_history[1:], "system")
    assert [message.role for message in projection.messages] == [
        "system", "user", "assistant", "tool", "assistant",
    ]
    assert projection.messages[2].tool_calls[0].call_id == projection.messages[3].call_id
    assert projection.consumed_call_ids == (projection.messages[3].call_id,)


def test_agent_can_finish_turn_with_session_metadata_in_history(tmp_path):
    class FakeClient:
        def complete(self, request):
            assert [message.role for message in request.messages] == ["system", "user"]
            return ModelReply(content="完成", finish_reason="stop")

    writer = new_writer(JsonlEventStore(tmp_path / "logs"))
    writer.append("session_created", create_payload(tmp_path))
    agent = AgentLoop(
        FakeClient(), ToolRegistry(), Workspace(tmp_path), writer, ContextBuilder(), LoopLimits(),
    )
    outcome = agent.run_turn("测试")
    assert (outcome.status, outcome.reason) == ("completed", "model_stop")
    assert [event.type for event in writer.history] == [
        "session_created", "user_message", "assistant_message", "turn_finished",
    ]


def test_reads_missing_and_empty_file(tmp_path):
    store = JsonlEventStore(tmp_path)
    session_id = uuid4()
    assert list(store.iter_events(session_id)) == []
    path = log_path(tmp_path, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert list(store.iter_events(session_id)) == []


@pytest.mark.parametrize("defect", [
    "invalid_json", "blank", "invalid_utf8", "version", "bool_version",
    "sequence_repeat", "sequence_gap", "duplicate_id", "foreign_session",
    "late_session_created", "missing_call_id", "non_utc", "epoch_as_string",
])
def test_rejects_corrupt_middle_record_and_preserves_bytes(tmp_path, memory_history, defect):
    session_id = memory_history[0].session_id
    store = JsonlEventStore(tmp_path)
    path = log_path(tmp_path, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = 2 if defect == "missing_call_id" else 1
    rows = [(item.model_dump_json() + "\n").encode("utf-8") for item in memory_history]
    bad_event = json.loads(rows[index])
    if defect == "version":
        bad_event["v"] = 2
    elif defect == "bool_version":
        bad_event["v"] = True
    elif defect == "sequence_repeat":
        bad_event["seq"] = 1
    elif defect == "sequence_gap":
        bad_event["seq"] = 3
    elif defect == "duplicate_id":
        bad_event["event_id"] = str(memory_history[0].event_id)
    elif defect == "foreign_session":
        bad_event["session_id"] = str(uuid4())
    elif defect == "late_session_created":
        bad_event["type"] = "session_created"
        bad_event["payload"] = memory_history[0].payload.model_dump(mode="json")
    elif defect == "missing_call_id":
        del bad_event["payload"]["tool_calls"][0]["call_id"]
    elif defect == "non_utc":
        bad_event["ts"] = "2026-11-20T12:30:00+08:00"
    elif defect == "epoch_as_string":
        bad_event["ts"] = "1700000000"
    corrupt_lines = {"invalid_json": b"{bad\n", "blank": b"\n", "invalid_utf8": b"\xff\n"}
    rows[index] = corrupt_lines.get(defect, (json.dumps(bad_event) + "\n").encode("utf-8"))
    original = b"".join(rows)
    path.write_bytes(original)
    for operation in (lambda: list(store.iter_events(session_id)),
                      lambda: store.append(memory_history[-1])):
        with pytest.raises(EventStoreCorruptError) as caught:
            operation()
        assert (caught.value.session_id, caught.value.line_number, caught.value.last_valid_seq) == (
            session_id, index + 1, index,
        )
        assert path.read_bytes() == original


@pytest.mark.parametrize("complete_json", [False, True])
def test_unterminated_tail_is_not_committed(tmp_path, memory_history, complete_json):
    store = JsonlEventStore(tmp_path)
    header, second, *_ = memory_history
    store.append(header)
    path = log_path(tmp_path, header.session_id)
    tail = second.model_dump_json().encode("utf-8") if complete_json else b'{"v":1,"payload":'
    raw = path.read_bytes() + tail
    path.write_bytes(raw)
    for operation in (lambda: list(store.iter_events(header.session_id)), lambda: store.append(second)):
        with pytest.raises(EventStoreCorruptError) as caught:
            operation()
        assert caught.value.line_number == 2
        assert caught.value.last_valid_seq == 1
        assert path.read_bytes() == raw


def test_rejects_duplicate_append_and_requires_creation_header(tmp_path, memory_history):
    store = JsonlEventStore(tmp_path)
    header, second, *_ = memory_history
    with pytest.raises(EventStoreError):
        store.append(second.model_copy(update={"seq": 1}))
    path = log_path(tmp_path, header.session_id)
    assert not path.exists() or path.read_bytes() == b""
    store.append(header)
    original = path.read_bytes()
    with pytest.raises(EventStoreError):
        store.append(header)
    with pytest.raises(EventStoreError):
        store.append(second.model_copy(update={"event_id": header.event_id}))
    with pytest.raises(EventStoreError):
        store.append(second.model_copy(update={"seq": 3}))
    assert path.read_bytes() == original
    store.append(second)
    assert list(store.iter_events(header.session_id)) == [header, second]


def test_publish_rejects_invalid_history_or_existing_target(tmp_path, memory_history):
    store = JsonlEventStore(tmp_path)
    header, second, *_ = memory_history
    path = log_path(tmp_path, header.session_id)
    with pytest.raises(EventStoreError):
        store.publish(header.session_id, [header, second.model_copy(update={"seq": 3})])
    assert not path.exists()
    store.publish(header.session_id, memory_history)
    original = path.read_bytes()
    with pytest.raises(EventStoreError):
        store.publish(header.session_id, [header])
    assert path.read_bytes() == original


def test_append_open_failure_is_reported_without_notifying(tmp_path, monkeypatch):
    store = JsonlEventStore(tmp_path)
    notices = []
    writer = new_writer(store, on_event=notices.append)
    header = writer.append("session_created", create_payload(tmp_path))
    path = log_path(tmp_path, header.session_id)
    before = path.read_bytes()
    original_open = Path.open

    def fail_on_write(target, mode="r", *args, **kwargs):
        if target == path and any(flag in mode for flag in "wax+"):
            raise PermissionError("test append access denial")
        return original_open(target, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_on_write)
        with pytest.raises(EventWriteError) as caught:
            writer.append("user_message", UserMessagePayload(turn_id=uuid4(), content="inspect"))
        assert isinstance(caught.value.__cause__, EventStoreError)
    assert notices == [header]
    assert path.read_bytes() == before
    assert new_writer(store, header.session_id).history == (header,)


def test_fsync_failure_returns_error_but_does_not_rewrite_committed_bytes(tmp_path, monkeypatch):
    notices = []
    store = JsonlEventStore(tmp_path)
    writer = new_writer(store, on_event=notices.append)
    header = writer.append("session_created", create_payload(tmp_path))

    def fail_fsync(_fd):
        raise OSError("test fsync failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_fsync)
        with pytest.raises(EventWriteError) as caught:
            writer.append("user_message", UserMessagePayload(turn_id=uuid4(), content="inspect"))
        assert isinstance(caught.value.__cause__, EventStoreError)
    assert notices == [header]

    reopened = JsonlEventStore(tmp_path)
    events = list(reopened.iter_events(header.session_id))
    assert events[0] == header
    assert events[1].payload.content == "inspect"
    with pytest.raises(EventStoreError):
        reopened.append(events[1])
    assert list(reopened.iter_events(header.session_id)) == events


def test_failed_partial_write_preserves_prefix_and_reports_torn_tail(
    tmp_path, memory_history, monkeypatch,
):
    store = JsonlEventStore(tmp_path)
    header, second, *_ = memory_history
    store.append(header)
    path = log_path(tmp_path, header.session_id)
    prefix = path.read_bytes()

    def fail_after_prefix(file, line):
        file.write(line[:12])
        raise OSError("test partial write")

    with monkeypatch.context() as patch:
        patch.setattr(event_store, "_write_line", fail_after_prefix)
        with pytest.raises(EventStoreError, match="Reread"):
            store.append(second)
    interrupted = path.read_bytes()
    assert interrupted.startswith(prefix)
    assert len(interrupted) == len(prefix) + 12
    with pytest.raises(EventStoreCorruptError, match="Unterminated") as caught:
        list(JsonlEventStore(tmp_path).iter_events(header.session_id))
    assert (caught.value.line_number, caught.value.last_valid_seq) == (2, 1)
    assert path.read_bytes() == interrupted


def child_env():
    names = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP",
             "USERPROFILE", "LOCALAPPDATA", "APPDATA", "SYSTEMDRIVE")
    env = {name: os.environ[name] for name in names if name in os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return env


def test_can_read_completed_history_in_another_process(tmp_path, memory_history):
    session_id = memory_history[0].session_id
    JsonlEventStore(tmp_path).publish(session_id, memory_history)
    script = """
from pathlib import Path
import sys
from uuid import UUID
from minicodeagent.context.store import JsonlEventStore

for event in JsonlEventStore(Path(sys.argv[1])).iter_events(UUID(sys.argv[2])):
    print(event.model_dump_json())
"""
    result = subprocess.run(
        [sys.executable, "-P", "-B", "-X", "utf8", "-c", script, str(tmp_path), str(session_id)],
        cwd=tmp_path, env=child_env(), check=True, capture_output=True, timeout=20,
    )
    events = tuple(SessionEvent.model_validate_json(line) for line in result.stdout.splitlines())
    assert events == memory_history


def test_process_holding_append_lock_blocks_same_session_only(tmp_path, memory_history):
    store = JsonlEventStore(tmp_path)
    header, second, *_ = memory_history
    store.append(header)
    ready = tmp_path / "writer-ready"
    script = """
import os
from pathlib import Path
import sys
from minicodeagent.context.events import SessionEvent
from minicodeagent.context.store import JsonlEventStore

fsync = os.fsync

def wait_with_lock(fd):
    Path(sys.argv[2]).touch()
    if not sys.stdin.readline():
        raise RuntimeError("parent terminated before releasing writer")
    fsync(fd)

os.fsync = wait_with_lock
JsonlEventStore(Path(sys.argv[1])).append(SessionEvent.model_validate_json(sys.argv[3]))
"""
    child = subprocess.Popen(
        [sys.executable, "-P", "-B", "-X", "utf8", "-c", script, str(tmp_path), str(ready),
         second.model_dump_json()],
        cwd=tmp_path, env=child_env(), stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        limit = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < limit:
            time.sleep(0.02)
        if child.poll() is not None:
            pytest.fail(child.communicate()[1].decode("utf-8", errors="replace"))
        assert ready.exists(), "child did not reach fsync with the session lock held"
        with pytest.raises(EventStoreBusyError):
            store.append(second)
        other = new_writer(store).append("session_created", create_payload(tmp_path))
        assert list(store.iter_events(other.session_id)) == [other]
        _, error = child.communicate(input=b"release\n", timeout=15)
        assert child.returncode == 0, error.decode("utf-8", errors="replace")
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert list(store.iter_events(header.session_id)) == [header, second]
