import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest

from minicodeagent.context.events import (
    AssistantMessagePayload,
    ToolResultPayload,
    TurnFinishedPayload,
    UserMessagePayload,
)
from minicodeagent.context.store import EventStoreBusyError, EventStoreCorruptError, EventStoreError, JsonlEventStore
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import ResumeError, SessionService
from minicodeagent.tools.registry import ToolRegistry


@pytest.fixture
def session_setup(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    sentinel = workspace / "sentinel.txt"
    sentinel.write_text("preexisting evidence", encoding="utf-8")
    store = JsonlEventStore(tmp_path / "data")
    return SessionService(store, system_prompt="Instructions"), store, workspace


def log_path(store, session_id):
    return store.root / "sessions" / f"{session_id}.jsonl"


def begin_turn(handle, content="Repair the fixture"):
    turn_id = uuid4()
    handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content=content))
    return turn_id


def record_calls(handle, turn_id, *calls):
    return handle.writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content="", finish_reason="tool_calls", tool_calls=calls,
    ))


def record_reply(handle, turn_id, content="Verified previous work", *, consumed=()):
    return handle.writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content=content, finish_reason="stop",
        consumed_call_ids=consumed,
    ))


def tool_call(provider_id):
    return ToolCall(
        provider_call_id=provider_id, name="write_file",
        arguments_raw='{"path":"sentinel.txt","content":"do not execute"}',
    )


def test_resume_completed_session_keeps_history_and_allows_append(session_setup):
    service, store, workspace = session_setup
    with service.create(workspace, "Persistent") as original:
        session_id = original.session_id
        turn_id = begin_turn(original)
        reply = record_reply(original, turn_id)
        original.writer.append("turn_finished", TurnFinishedPayload(
            turn_id=turn_id, status="completed", reason="model_stop", assistant_seq=reply.seq,
        ))
        prior = original.state.history
    path = log_path(store, session_id)
    before = path.read_bytes()

    with service.resume(session_id) as resumed:
        assert resumed.session_id == session_id
        assert resumed.title == "Persistent"
        assert resumed.workspace.root == workspace.resolve()
        assert resumed.state.history == prior
        assert path.read_bytes() == before
        new_turn = begin_turn(resumed, "Check a new task")
        assert resumed.state.current_turn.turn_id == new_turn
        assert resumed.state.history[-1].seq == prior[-1].seq + 1
        assert sum(e.type == "session_created" for e in resumed.state.history) == 1


def test_resume_recovers_saved_terminal_reply_without_duplicating_it(session_setup):
    service, _, workspace = session_setup
    with service.create(workspace) as original:
        session_id = original.session_id
        turn_id = begin_turn(original)
        reply = record_reply(original, turn_id)

    with service.resume(session_id) as resumed:
        history = resumed.state.history
        assert [event.type for event in history] == [
            "session_created", "user_message", "assistant_message", "turn_finished",
        ]
        assert history[-2] == reply
        assert resumed.state.current_turn.finished == TurnFinishedPayload(
            turn_id=turn_id, status="completed", reason="model_stop", assistant_seq=reply.seq,
        )
    with service.resume(session_id) as reopened:
        assert reopened.state.history == history


def test_resume_asks_before_marking_unknown_and_only_settles_pending(session_setup, monkeypatch):
    service, store, workspace = session_setup
    execute = Mock(side_effect=AssertionError("Historical calls must not be dispatched during recovery."))
    monkeypatch.setattr(ToolRegistry, "execute", execute)
    skipped, unknown, first, second = (tool_call(name) for name in ("skip", "unknown", "first", "second"))
    with service.create(workspace) as original:
        session_id = original.session_id
        turn_id = begin_turn(original)
        record_calls(original, turn_id, skipped, unknown, first, second)
        known = tuple(
            original.writer.append("tool_result", ToolResultPayload(
                turn_id=turn_id, call_id=call.call_id, status="interrupted",
                output=note, details={"execution_outcome": outcome},
            ))
            for call, note, outcome in (
                (skipped, "Not dispatched.", "not_started"),
                (unknown, "Stopped before observing result.", "unknown"),
            )
        )
    path = log_path(store, session_id)
    before = path.read_bytes()
    with pytest.raises(ResumeError):
        service.resume(session_id)
    asked = []

    def decline(calls):
        asked.append(calls)
        return False

    with pytest.raises(ResumeError):
        service.resume(session_id, confirm_pending=decline)
    assert asked == [(first, second)]
    assert path.read_bytes() == before
    execute.assert_not_called()
    with service.open(session_id) as opened:
        assert opened.replay_result.pending_calls == (first, second)
        assert opened.replay_result.unknown_call_ids == {unknown.call_id, first.call_id, second.call_id}
        assert opened.replay_result.projection is None

    def confirm(calls):
        assert calls == (first, second)
        return True

    with service.resume(session_id, confirm_pending=confirm) as resumed:
        history = resumed.state.history
        assert tuple(event for event in history if event.type == "tool_result")[:2] == known
        results = [event.payload for event in history if event.type == "tool_result"]
        assert [result.call_id for result in results] == [
            skipped.call_id, unknown.call_id, first.call_id, second.call_id,
        ]
        assert all(result.status == "interrupted" for result in results)
        assert [result.details["execution_outcome"] for result in results] == [
            "not_started", "unknown", "unknown", "unknown",
        ]
        projection = resumed.replay_result
        assert projection.pending_calls == ()
        assert projection.unknown_call_ids == {unknown.call_id, first.call_id, second.call_id}
        assert resumed.state.current_turn.finished.status == "interrupted"
        assert projection.projection is not None
    with service.resume(session_id) as reopened:
        assert reopened.state.history == history
    execute.assert_not_called()
    assert (workspace / "sentinel.txt").read_text(encoding="utf-8") == "preexisting evidence"


def test_legacy_unfinished_turn_can_close_but_cannot_continue_blindly(session_setup):
    service, store, workspace = session_setup
    with service.create(workspace) as original:
        old_turn = begin_turn(original)
        session_id = original.session_id
    path = log_path(store, session_id)
    before = path.read_bytes()
    with pytest.raises(ResumeError):
        service.resume(session_id, continue_turn=True)
    assert path.read_bytes() == before
    with service.resume(session_id) as resumed:
        assert resumed.state.current_turn.turn_id == old_turn
        assert resumed.state.current_turn.finished.status == "interrupted"
        assert begin_turn(resumed) != old_turn


def test_resume_continues_registered_turn_without_assuming_unanswered_request_was_unsent(session_setup):
    from minicodeagent.context.events import ModelRequestStartedPayload

    service, store, workspace = session_setup
    with service.create(workspace) as original:
        turn_id = begin_turn(original)
        session_id = original.session_id
        request = original.writer.append("model_request_started", ModelRequestStartedPayload(
            turn_id=turn_id, request_id=uuid4(), attempt=1, trigger="initial",
        ))
    path = log_path(store, session_id)
    before = path.read_bytes()
    with service.resume(session_id, continue_turn=True) as resumed:
        assert resumed.state.current_turn.turn_id == turn_id
        assert resumed.state.current_turn.finished is None
        assert resumed.state.history[-1] == request
        assert resumed.replay_result.projection is not None
        assert path.read_bytes() == before


def test_busy_and_corrupt_sources_fail_for_both_resume_and_fork(session_setup):
    service, store, workspace = session_setup
    with service.create(workspace) as original:
        session_id = original.session_id
        for operation in (service.resume, service.fork):
            with pytest.raises(EventStoreBusyError):
                operation(session_id)
    path = log_path(store, session_id)
    damaged = path.read_bytes() + b"not json\n"
    path.write_bytes(damaged)
    for operation in (service.resume, service.fork):
        with pytest.raises(EventStoreCorruptError):
            operation(session_id)
    assert list(path.parent.glob("*.jsonl")) == [path]
    assert path.read_bytes() == damaged


def test_fork_preserves_evidence_and_rekeys_envelopes_under_an_independent_lock(session_setup):
    service, store, workspace = session_setup
    with service.create(workspace, "Parent") as original:
        parent_id = original.session_id
        turn_id = begin_turn(original)
        call = tool_call("old-call")
        record_calls(original, turn_id, call)
        original.writer.append("tool_result", ToolResultPayload(
            turn_id=turn_id, call_id=call.call_id, status="ok", output="Existing result",
        ))
        reply = record_reply(original, turn_id, consumed=(call.call_id,))
        original.writer.append("turn_finished", TurnFinishedPayload(
            turn_id=turn_id, status="completed", reason="model_stop", assistant_seq=reply.seq,
        ))
        parent_history = original.state.history
    parent_path = log_path(store, parent_id)
    before = parent_path.read_bytes()

    with service.fork(parent_id, title="Child") as child:
        child_id = child.session_id
        assert child_id != parent_id
        assert child.title == "Child"
        assert child.workspace.root == workspace.resolve()
        history = child.state.history
        assert [event.seq for event in history] == list(range(1, len(parent_history) + 1))
        assert sum(event.type == "session_created" for event in history) == 1
        assert all(event.session_id == child_id for event in history)
        assert {event.event_id for event in history}.isdisjoint(event.event_id for event in parent_history)
        origin = child.state.metadata
        assert origin.parent_session_id == parent_id and origin.parent_through_seq == parent_history[-1].seq
        for inherited, source in zip(history[1:], parent_history[1:], strict=True):
            assert (inherited.type, inherited.payload, inherited.seq, inherited.ts) == (
                source.type, source.payload, source.seq, source.ts,
            )
        assert child.replay_result.consumed_call_ids == {call.call_id}
        with service.open(parent_id) as independently_open:
            assert independently_open.state.history == parent_history
            with pytest.raises(EventStoreBusyError):
                service.open(child_id)
            child_turn = begin_turn(child, "Diverge")
        assert parent_path.read_bytes() == before
    with service.open(child_id) as reopened:
        assert reopened.state.current_turn.turn_id == child_turn
        assert reopened.state.last_seq == len(parent_history) + 1
        assert reopened.replay_result.projection is not None


def test_fork_from_pending_history_keeps_pending_and_binds_explicit_workspace(session_setup, tmp_path):
    service, store, workspace = session_setup
    override = tmp_path / "other-project"
    override.mkdir()
    with service.create(workspace) as original:
        parent_id = original.session_id
        turn_id = begin_turn(original)
        call = tool_call("unknown")
        record_calls(original, turn_id, call)
    path = log_path(store, parent_id)
    before = path.read_bytes()
    with service.fork(parent_id, workspace=override) as child:
        assert child.workspace.root == override.resolve()
        assert child.state.metadata.workspace == str(override.resolve())
        assert child.replay_result.pending_calls == (call,)
        assert child.replay_result.unknown_call_ids == {call.call_id}
        assert child.replay_result.projection is None
    with service.open(parent_id) as parent:
        assert parent.workspace.root == workspace.resolve()
    assert path.read_bytes() == before


def test_failed_fork_does_not_publish_a_candidate_or_retain_locks(session_setup, monkeypatch):
    service, store, workspace = session_setup
    with service.create(workspace) as original:
        parent_id = original.session_id
    parent_path = log_path(store, parent_id)
    before = parent_path.read_bytes()
    attempts = []
    original_replace = os.replace

    def fail_child_publish(source, destination):
        destination = Path(destination)
        if destination.suffix == ".jsonl":
            attempts.append(UUID(destination.stem))
            raise OSError("simulated publish failure")
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_child_publish)
        with pytest.raises(EventStoreError):
            service.fork(parent_id)
    assert len(attempts) == 1 and attempts[0] != parent_id
    assert list(parent_path.parent.iterdir()) == [parent_path]
    assert parent_path.read_bytes() == before
    with store.lock_session(attempts[0]) as lease:
        assert not tuple(lease.iter_events(attempts[0]))
    with service.open(parent_id):
        assert [row.id for row in service.list()] == [parent_id]


def test_fork_reopen_failure_identifies_committed_child(session_setup, monkeypatch):
    service, store, workspace = session_setup
    with service.create(workspace) as original:
        parent_id = original.session_id
        turn_id = begin_turn(original, "Persist the branch point")
    path = log_path(store, parent_id)
    before = path.read_bytes()
    child_ids = []

    def unavailable(lease, _workspace):
        child_ids.append(lease.session_id)
        raise EventStoreError("Temporary read failure")

    with monkeypatch.context() as patch:
        patch.setattr(service, "_open_locked", unavailable)
        with pytest.raises(EventStoreError, match="was published") as failure:
            service.fork(parent_id)
    child_id, = child_ids
    assert str(child_id) in str(failure.value)
    assert path.read_bytes() == before
    with service.open(child_id) as child:
        assert child.state.metadata.parent_session_id == parent_id
        assert child.state.current_turn.turn_id == turn_id


def run_session_process(code, *args, cwd, returncode=0):
    system_keys = (
        "SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP",
        "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "SYSTEMDRIVE",
    )
    env = {key: os.environ[key] for key in system_keys if key in os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    process = subprocess.run(
        [sys.executable, "-B", "-c", code, *(str(arg) for arg in args)],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert process.returncode == returncode, process.stderr
    return process.stdout.strip()


@pytest.mark.parametrize("crash", [False, True], ids=["close", "abrupt-exit"])
def test_resume_in_separate_process_after_partial_turn(tmp_path, session_setup, crash):
    service, store, workspace = session_setup
    first = """
import os
from pathlib import Path
import sys
from uuid import uuid4

from minicodeagent.context.events import AssistantMessagePayload, UserMessagePayload
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionService

service = SessionService(JsonlEventStore(Path(sys.argv[1])))
workspace = Path(sys.argv[2])
with service.create(workspace, "Between processes") as handle:
    turn_id = uuid4()
    handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="Carry on"))
    call = ToolCall(
        provider_call_id="workspace-write", name="write_file",
        arguments_raw='{"path":"sentinel.txt","content":"would be a duplicate effect"}',
    )
    handle.writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content="", finish_reason="tool_calls",
        tool_calls=(call,),
    ))
    # Model a successful external effect whose result event was never persisted.
    (workspace / "sentinel.txt").write_text("effect already occurred", encoding="utf-8")
    print(handle.session_id, call.call_id, flush=True)
    if sys.argv[3] == "crash":
        os._exit(23)
"""
    session_id, call_id = map(UUID, run_session_process(
        first, store.root, workspace, "crash" if crash else "close", cwd=tmp_path,
        returncode=23 if crash else 0,
    ).split())
    path = log_path(store, session_id)
    before = path.read_bytes()
    second = """
from pathlib import Path
import sys
from uuid import UUID, uuid4

from minicodeagent.context.events import UserMessagePayload
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.session.service import SessionService

service = SessionService(JsonlEventStore(Path(sys.argv[1])))
session_id = UUID(sys.argv[2])
call_id = UUID(sys.argv[3])
with service.open(session_id) as opened:
    assert [call.call_id for call in opened.replay_result.pending_calls] == [call_id]
    assert opened.replay_result.unknown_call_ids == {call_id}
def confirm(calls):
    assert [call.call_id for call in calls] == [call_id]
    assert calls[0].provider_call_id == "workspace-write"
    return True
with service.resume(session_id, confirm_pending=confirm) as resumed:
    assert resumed.state.current_turn.finished.status == "interrupted"
    assert resumed.replay_result.pending_calls == ()
    assert resumed.replay_result.unknown_call_ids == {call_id}
    assert resumed.state.history[-2].payload.details["execution_outcome"] == "unknown"
    resumed.writer.append("user_message", UserMessagePayload(turn_id=uuid4(), content="A new task"))
    print(resumed.state.last_seq)
"""
    assert run_session_process(second, store.root, session_id, call_id, cwd=tmp_path) == "6"
    assert path.read_bytes().startswith(before)
    with service.open(session_id) as reopened:
        assert reopened.state.history[-1].payload.content == "A new task"
        assert tuple(event.payload.call_id for event in reopened.state.history if event.type == "tool_result") == (
            call_id,
        )
    assert (workspace / "sentinel.txt").read_text(encoding="utf-8") == "effect already occurred"


def test_forked_session_opens_independently_in_another_process(tmp_path, session_setup):
    service, store, workspace = session_setup
    with service.create(workspace) as original:
        parent_id = original.session_id
        turn_id = begin_turn(original, "Original task")
        reply = record_reply(original, turn_id)
        original.writer.append("turn_finished", TurnFinishedPayload(
            turn_id=turn_id, status="completed", reason="model_stop", assistant_seq=reply.seq,
        ))
    parent_path = log_path(store, parent_id)
    before = parent_path.read_bytes()
    first = """
from pathlib import Path
import sys
from uuid import UUID
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.session.service import SessionService

service = SessionService(JsonlEventStore(Path(sys.argv[1])))
with service.fork(UUID(sys.argv[2]), title="Child from separate process") as child:
    print(child.session_id)
"""
    child_id = UUID(run_session_process(first, store.root, parent_id, cwd=tmp_path))
    second = """
from pathlib import Path
import sys
from uuid import UUID, uuid4

from minicodeagent.context.events import UserMessagePayload
from minicodeagent.context.store import EventStoreBusyError, JsonlEventStore
from minicodeagent.session.service import SessionService

service = SessionService(JsonlEventStore(Path(sys.argv[1])))
parent_id, child_id = map(UUID, sys.argv[2:])
with service.open(parent_id) as parent, service.open(child_id) as child:
    assert parent.state.last_seq == child.state.last_seq == 4
    assert child.state.metadata.parent_session_id == parent_id
    for session_id in (parent_id, child_id):
        try:
            service.open(session_id)
        except EventStoreBusyError:
            pass
        else:
            raise AssertionError("The handle's lock did not block a duplicate open")
    child.writer.append("user_message", UserMessagePayload(turn_id=uuid4(), content="Child only"))
    assert child.state.last_seq == 5 and parent.state.last_seq == 4
"""
    assert run_session_process(second, store.root, parent_id, child_id, cwd=tmp_path) == ""
    assert parent_path.read_bytes() == before
    with service.open(child_id) as reopened:
        assert reopened.state.history[-1].payload.content == "Child only"
