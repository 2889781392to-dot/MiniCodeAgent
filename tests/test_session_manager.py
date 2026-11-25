from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import UUID, uuid4

import pytest

from minicodeagent.context.events import (
    AssistantMessagePayload,
    SessionCreatedPayload,
    SessionEvent,
    TurnFinishedPayload,
    UserMessagePayload,
)
from minicodeagent.context.store import (
    EventStoreBusyError,
    EventStoreCorruptError,
    EventStoreError,
    JsonlEventStore,
)
from minicodeagent.context.writer import EventWriteError
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionHandle, SessionService
from minicodeagent.workspace import WorkspaceError


@pytest.fixture
def managed(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "keep.txt").write_bytes(b"leave this file alone")
    store = JsonlEventStore(tmp_path / "data")
    return SessionService(store, system_prompt="Project instructions"), store, workspace


def log_path(store, session_id):
    return store.root / "sessions" / f"{session_id}.jsonl"


def begin_turn(handle):
    turn_id = uuid4()
    return handle.writer.append("user_message", UserMessagePayload(
        turn_id=turn_id, content="Inspect the code",
    ))


def test_create_publishes_one_header_for_each_unique_session(managed):
    service, store, workspace = managed
    with service.create(workspace / ".." / workspace.name, title="First session") as first:
        with service.new(workspace, title="Second session") as second:
            assert isinstance(first, SessionHandle)
            assert first.session_id != second.session_id
            for handle, title in ((first, "First session"), (second, "Second session")):
                assert handle.workspace.root == workspace.resolve()
                assert handle.title == title
                assert not handle.closed
                events = tuple(handle.store.iter_events(handle.session_id, require_log=True))
                assert events == handle.writer.history == handle.state.history
                assert len(events) == len(log_path(store, handle.session_id).read_bytes().splitlines()) == 1
                assert (events[0].type, events[0].seq, events[0].session_id) == (
                    "session_created", 1, handle.session_id,
                )
                assert events[0].payload == SessionCreatedPayload(title=title, workspace=str(workspace.resolve()))
                assert handle.replay_result.projection.messages[0].content == "Project instructions"
                assert handle.state.metadata == events[0].payload
    assert first.closed and second.closed


def test_failed_create_leaves_no_published_session_or_staged_log(managed, monkeypatch):
    service, store, workspace = managed
    original_replace = os.replace
    attempts = []

    def fail_publish(source, target):
        target = Path(target)
        if target.suffix == ".jsonl":
            attempts.append(UUID(target.stem))
            raise OSError("simulated failure publishing session")
        return original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_publish)
        with pytest.raises(EventStoreError):
            service.create(workspace, title="Not published")

    assert not list((store.root / "sessions").glob("*"))
    assert service.list() == []
    assert len(attempts) == 1
    with store.lock_session(attempts[0]) as lease:
        assert tuple(lease.iter_events(attempts[0])) == ()
    with service.create(workspace, title="Published after failure") as handle:
        session_id = handle.session_id
    assert [record.id for record in service.list()] == [session_id]


def test_catalog_empty_and_then_uses_event_metadata_and_log_order(managed):
    service, store, workspace = managed
    assert service.list() == []
    with service.create(workspace, title="First") as first:
        pass
    with service.create(workspace, title="Second") as second:
        pass
    with service.open(first.session_id) as handle:
        new_event = begin_turn(handle)

    rows = service.list()
    assert [row.id for row in rows] == [first.session_id, second.session_id]
    assert [row.title for row in rows] == ["First", "Second"]
    assert {row.status for row in rows} == {"valid"}
    assert {row.workspace for row in rows} == {str(workspace.resolve())}
    assert rows[0].last_seq == new_event.seq
    assert rows[0].updated_at == new_event.ts
    assert rows[0].turn_status == "open"
    assert (store.root / "session_index.json").is_file()
    assert [path.name for path in workspace.iterdir()] == ["keep.txt"]
    assert (workspace / "keep.txt").read_bytes() == b"leave this file alone"


def test_catalog_reports_corruption_busy_and_rebuilds_invalid_index(managed):
    service, store, workspace = managed
    handles = []
    for title in ("Healthy", "Will be busy", "Will be damaged"):
        with service.create(workspace, title=title) as handle:
            handles.append(handle)
    healthy, busy, damaged = handles
    original = service.list()
    assert {row.status for row in original} == {"valid"}
    damaged_path = log_path(store, damaged.session_id)
    bad_bytes = damaged_path.read_bytes() + b"{corrupt-json}\n"
    damaged_path.write_bytes(bad_bytes)
    invalid_name = damaged_path.parent / "unrecognized.jsonl"
    invalid_name.write_bytes(b"x\n")
    staged = damaged_path.parent / "incomplete.tmp"
    staged.write_bytes(b"x")

    with service.open(busy.session_id):
        rows = {row.log_name: row for row in service.list()}
        assert rows[log_path(store, healthy.session_id).name].status == "valid"
        locked = rows[log_path(store, busy.session_id).name]
        assert (locked.status, locked.title, locked.workspace) == (
            "busy", "Will be busy", str(workspace.resolve()),
        )
        invalid = (rows[damaged_path.name], rows[invalid_name.name])
        assert all(row.status == "corrupt" and row.error for row in invalid)
        assert staged.name not in rows
    assert damaged_path.read_bytes() == bad_bytes

    index_path = store.root / "session_index.json"
    index_path.write_bytes(b"broken cache")
    rebuilt = {row.log_name: row for row in service.list()}
    assert rebuilt[log_path(store, busy.session_id).name].status == "valid"
    assert rebuilt[damaged_path.name].status == "corrupt"
    assert index_path.read_bytes() != b"broken cache"


def test_catalog_breaks_equal_timestamps_by_id_and_uses_recorded_workspace(managed, tmp_path):
    service, store, _ = managed
    absent = tmp_path / "workspace-no-longer-present"
    when = datetime(2025, 1, 1, tzinfo=UTC)
    session_ids = (UUID(int=1), UUID(int=2))
    for session_id in session_ids:
        store.append(SessionEvent(
            session_id=session_id, event_id=uuid4(), seq=1, ts=when, type="session_created",
            payload=SessionCreatedPayload(title="Stored title", workspace=str(absent)),
        ))

    rows = service.list()
    assert [row.id for row in rows] == list(reversed(session_ids))
    assert all(row.status == "valid" and row.workspace == str(absent) for row in rows)
    assert not absent.exists()
    with pytest.raises(WorkspaceError):
        service.open(session_ids[0])


def test_handle_uses_same_lock_for_reads_writes_and_close_invalidates_references(managed):
    service, store, workspace = managed
    handle = service.create(workspace)
    writer, leased_store, session_id = handle.writer, handle.store, handle.session_id
    try:
        user_event = begin_turn(handle)
        assert handle.replay_result.state.history == writer.history
        assert tuple(leased_store.iter_events(session_id))[-1] == user_event
        with pytest.raises(EventStoreBusyError):
            service.open(session_id)
        with pytest.raises(EventStoreBusyError):
            list(JsonlEventStore(store.root).iter_events(session_id))
        with pytest.raises(EventStoreError):
            list(leased_store.iter_events(uuid4()))
        with pytest.raises(EventStoreError):
            leased_store.append(user_event.model_copy(update={"session_id": uuid4()}))
        before = log_path(store, session_id).read_bytes()
    finally:
        service.close(handle)

    assert handle.closed
    handle.close()
    valid_payload = AssistantMessagePayload(
        turn_id=user_event.payload.turn_id, request_id=uuid4(), content="Continuation", finish_reason="stop",
    )
    valid_event = SessionEvent(
        session_id=session_id, event_id=uuid4(), seq=user_event.seq + 1, ts=user_event.ts,
        type="assistant_message", payload=valid_payload,
    )
    with pytest.raises(EventWriteError):
        writer.append("assistant_message", valid_payload)
    with pytest.raises(EventStoreError):
        leased_store.append(valid_event)
    assert log_path(store, session_id).read_bytes() == before
    with service.open(session_id) as reopened:
        assert not reopened.closed
        assert reopened.state.current_turn.turn_id == user_event.payload.turn_id


def test_open_override_preserves_persisted_workspace_and_failed_bind_releases_lock(managed, tmp_path):
    service, store, workspace = managed
    with service.create(workspace) as handle:
        session_id = handle.session_id
    path = log_path(store, session_id)
    before = path.read_bytes()
    with pytest.raises(WorkspaceError):
        service.open(session_id, workspace=workspace / "missing-directory")
    alternative = tmp_path / "different-project"
    alternative.mkdir()
    with service.open(session_id, workspace=alternative) as override:
        assert override.workspace.root == alternative.resolve()
        assert override.state.metadata.workspace == str(workspace.resolve())
    with service.open(session_id) as original:
        assert original.workspace.root == workspace.resolve()
    assert path.read_bytes() == before


def test_corrupt_open_releases_lease_without_repair(managed):
    service, store, workspace = managed
    with service.create(workspace) as handle:
        session_id = handle.session_id
    path = log_path(store, session_id)
    original = path.read_bytes()
    damaged = original + b"bad-json\n"
    path.write_bytes(damaged)
    for _ in range(2):
        with pytest.raises(EventStoreCorruptError):
            service.open(session_id)
    assert path.read_bytes() == damaged


def test_handle_context_releases_lock_when_body_raises(managed):
    service, _, workspace = managed
    with pytest.raises(RuntimeError, match="failed inside context"):
        with service.create(workspace) as handle:
            raise RuntimeError("failed inside context")
    assert handle.closed
    with service.open(handle.session_id) as reopened:
        assert reopened.state.last_seq == 1


def test_publish_refuses_iterable_that_closes_its_lease(managed):
    _, store, workspace = managed
    session_id = uuid4()
    header = SessionEvent(
        session_id=session_id, event_id=uuid4(), seq=1, ts=datetime(2025, 1, 1, tzinfo=UTC),
        type="session_created", payload=SessionCreatedPayload(title="New session", workspace=str(workspace)),
    )
    with store.lock_session(session_id) as lease:
        def closes_while_iterating():
            yield header
            lease.close()

        with pytest.raises(EventStoreError, match="closed"):
            lease.publish(session_id, closes_while_iterating())
        assert lease.closed

    assert not log_path(store, session_id).exists()
    with store.lock_session(session_id) as next_lease:
        next_lease.publish(session_id, [header])
        assert tuple(next_lease.iter_events(session_id)) == (header,)


def test_finished_turn_does_not_close_handle_and_pending_remains_unknown(managed):
    service, _, workspace = managed
    with service.create(workspace) as handle:
        session_id = handle.session_id
        turn_id = begin_turn(handle).payload.turn_id
        reply = handle.writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn_id, request_id=uuid4(), content="done", finish_reason="stop",
        ))
        terminal = handle.writer.append("turn_finished", TurnFinishedPayload(
            turn_id=turn_id, status="completed", reason="model_stop", assistant_seq=reply.seq,
        ))
        assert handle.state.current_turn.finished == terminal.payload
        assert not handle.closed
        next_turn = begin_turn(handle).payload.turn_id
        call = ToolCall(provider_call_id="write", name="write_file", arguments_raw='{"path":"keep.txt"}')
        handle.writer.append("assistant_message", AssistantMessagePayload(
            turn_id=next_turn, request_id=uuid4(), content="", finish_reason="tool_calls", tool_calls=(call,),
        ))

    with service.open(session_id) as reopened:
        assert not reopened.closed
        assert reopened.state.current_turn.turn_id == next_turn
        assert reopened.state.current_turn.finished is None
        result = reopened.replay_result
        assert result.pending_calls == (call,)
        assert result.unknown_call_ids == frozenset({call.call_id})
        assert result.consumed_call_ids == frozenset()
        assert result.projection is None
    assert (workspace / "keep.txt").read_bytes() == b"leave this file alone"


@pytest.mark.parametrize("crash", [False, True], ids=["close", "abrupt-exit"])
def test_other_process_owns_lock_until_close_or_abrupt_exit(managed, tmp_path, crash):
    service, store, workspace = managed
    with service.create(workspace, title="Shared session") as original:
        session_id = original.session_id
    ready = tmp_path / "child-ready"
    code = """
import os
from pathlib import Path
import sys
from uuid import UUID, uuid4

from minicodeagent.context.events import AssistantMessagePayload, UserMessagePayload
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionService

service = SessionService(JsonlEventStore(Path(sys.argv[1])), system_prompt="Child process")
with service.open(UUID(sys.argv[2])) as handle:
    Path(sys.argv[3]).touch(exist_ok=False)
    if sys.stdin.buffer.read(1) == b"x":
        turn_id = uuid4()
        handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="Inspect project"))
        call = ToolCall(provider_call_id="child-call", name="read_file", arguments_raw='{"path":"keep.txt"}')
        handle.writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn_id, request_id=uuid4(), content="", finish_reason="tool_calls", tool_calls=(call,),
        ))
        os._exit(23)
"""
    system_keys = (
        "SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP",
        "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "SYSTEMDRIVE",
    )
    env = {key: os.environ[key] for key in system_keys if key in os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", code, str(store.root), str(session_id), str(ready)],
        cwd=tmp_path, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if not ready.exists():
            if child.poll() is None:
                child.kill()
            _, errors = child.communicate(timeout=10)
            pytest.fail(f"Child never acquired the session lock: {errors.decode(errors='replace')}")
        assert child.poll() is None
        with pytest.raises(EventStoreBusyError):
            service.open(session_id)
        with pytest.raises(EventStoreBusyError):
            tuple(store.iter_events(session_id))
        row = next(row for row in service.list() if row.id == session_id)
        assert row.status == "busy" and row.title == "Shared session"

        with service.create(workspace, title="Independent session") as independent:
            assert independent.session_id != session_id
        _, errors = child.communicate(input=b"x" if crash else b"c", timeout=10)
        assert child.returncode == (23 if crash else 0), errors.decode(errors="replace")
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)

    with service.open(session_id) as reopened:
        assert reopened.title == "Shared session"
        assert reopened.state.last_seq == (3 if crash else 1)
        if crash:
            result = reopened.replay_result
            assert reopened.state.current_turn.finished is None
            assert len(result.pending_calls) == 1
            assert result.pending_calls[0].provider_call_id == "child-call"
            assert result.unknown_call_ids == {result.pending_calls[0].call_id}
            assert result.projection is None
