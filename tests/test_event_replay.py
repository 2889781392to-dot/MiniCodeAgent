from pathlib import Path
from uuid import uuid4

import pytest

from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.events import (
    AssistantMessagePayload,
    HistoryError,
    SessionCreatedPayload,
    ToolResultPayload,
    TurnFinishedPayload,
    UserMessagePayload,
)
from minicodeagent.context.replay import EmptySessionLogError, SessionReplay, SessionReplayResult
from minicodeagent.context.store import (
    EventLogNotFoundError,
    EventStoreBusyError,
    EventStoreCorruptError,
    JsonlEventSink,
    JsonlEventStore,
)
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ToolCall
from minicodeagent.tools.registry import ToolRegistry


SYSTEM_PROMPT = "Inspect the repository."


def log_path(root, session_id):
    return root / "sessions" / f"{session_id}.jsonl"


def snapshot(root):
    return {
        path.relative_to(root): (path.stat().st_mtime_ns, None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


@pytest.fixture
def session(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "app.py").write_text("answer = 42\n", encoding="utf-8")
    store = JsonlEventStore(tmp_path / "logs")
    session_id = uuid4()
    writer = SessionEventWriter(session_id=session_id, sink=JsonlEventSink(store, session_id))
    writer.append("session_created", SessionCreatedPayload(title="Persistent session", workspace=str(workspace)))
    return store, writer


def start_turn(writer, content="Inspect the code"):
    turn_id = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content=content))
    return turn_id


def call(provider_id):
    return ToolCall(provider_call_id=provider_id, name="read_file", arguments_raw='{"path":"app.py"}')


def reply(writer, turn_id, *, calls=(), consumed=()):
    return writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content="" if calls else "I inspected the file.",
        finish_reason="tool_calls" if calls else "stop", tool_calls=calls, consumed_call_ids=consumed,
    ))


def settle(writer, turn_id, tool_call, *, execution_outcome=None):
    status = "ok" if execution_outcome is None else "interrupted"
    return writer.append("tool_result", ToolResultPayload(
        turn_id=turn_id, call_id=tool_call.call_id, status=status, output="Recorded result",
        details={} if execution_outcome is None else {"execution_outcome": execution_outcome},
    ))


def finish(writer, turn_id, assistant, *, interrupted=False):
    return writer.append("turn_finished", TurnFinishedPayload(
        turn_id=turn_id, status="interrupted" if interrupted else "completed",
        reason="interrupted" if interrupted else "model_stop", assistant_seq=assistant.seq,
    ))


def diagnostic_codes(result):
    return {diagnostic.code for diagnostic in result.diagnostics}


def test_replays_multiple_turns_and_preserves_pairs_and_recorded_consumption(session):
    store, writer = session
    first_turn = start_turn(writer)
    first, second = call("first"), call("second")
    reply(writer, first_turn, calls=(first, second))
    settle(writer, first_turn, first)
    settle(writer, first_turn, second)
    final_reply = reply(writer, first_turn, consumed=(first.call_id, second.call_id))
    finish(writer, first_turn, final_reply)
    second_turn = start_turn(writer, "Give a short summary")
    summary = reply(writer, second_turn)
    terminal = finish(writer, second_turn, summary)
    original = writer.history

    result = SessionReplay(JsonlEventStore(store.root)).replay(writer.session_id, SYSTEM_PROMPT)

    assert isinstance(result, SessionReplayResult)
    assert result.state.history == original
    assert (result.state.session_id, result.state.metadata) == (writer.session_id, original[0].payload)
    assert result.state.current_turn.turn_id == second_turn
    assert result.state.current_turn.finished == terminal.payload
    assert (result.last_seq, result.next_seq) == (terminal.seq, terminal.seq + 1)
    assert result.pending_calls == () and result.unknown_call_ids == frozenset()
    assert result.consumed_call_ids == frozenset({first.call_id, second.call_id})
    assert result.projection == ContextBuilder().project(original, SYSTEM_PROMPT)
    assert [message.role for message in result.projection.messages] == [
        "system", "user", "assistant", "tool", "tool", "assistant", "user", "assistant",
    ]
    _, _, batch, result_one, result_two, *_ = result.projection.messages
    assert tuple(item.call_id for item in batch.tool_calls) == (result_one.call_id, result_two.call_id)
    assert result.diagnostics == ()


def test_pending_results_stay_unknown_without_building_or_synthesizing_messages(session):
    store, writer = session
    turn_id = start_turn(writer)
    settled, first, second = call("settled"), call("first"), call("second")
    reply(writer, turn_id, calls=(settled, first, second))
    logged_result = settle(writer, turn_id, settled)

    class ForbiddenBuilder(ContextBuilder):
        def project(self, history, system_prompt):
            pytest.fail("An unfinished tool batch reached ContextBuilder")

    result = SessionReplay(store, builder=ForbiddenBuilder()).replay(writer.session_id, SYSTEM_PROMPT)

    assert result.pending_calls == (first, second)
    assert result.unknown_call_ids == frozenset({first.call_id, second.call_id})
    assert result.consumed_call_ids == frozenset()
    assert result.state.current_turn.finished is None
    assert result.state.history == writer.history
    assert result.state.history[-1] == logged_result
    assert result.projection is None
    assert diagnostic_codes(result) == {"unfinished_turn", "pending_tool_results"}


def test_fully_paired_open_turn_projects_without_inventing_consumption_or_terminal(session):
    store, writer = session
    turn_id = start_turn(writer)
    read = call("read")
    reply(writer, turn_id, calls=(read,))
    settle(writer, turn_id, read)
    last_reply = reply(writer, turn_id)

    result = SessionReplay(store).replay(writer.session_id, SYSTEM_PROMPT)

    assert result.state.current_turn.latest_reply == last_reply
    assert result.state.current_turn.finished is None
    assert result.pending_calls == () and result.unknown_call_ids == frozenset()
    assert result.consumed_call_ids == frozenset()
    assert result.projection == ContextBuilder().project(writer.history, SYSTEM_PROMPT)
    assert result.projection.consumed_call_ids == (read.call_id,)
    assert diagnostic_codes(result) == {"unfinished_turn"}


def test_recorded_interruption_is_settled_and_preserves_execution_outcomes(session):
    store, writer = session
    turn_id = start_turn(writer)
    unknown, unstarted = call("unknown"), call("unstarted")
    batch = reply(writer, turn_id, calls=(unknown, unstarted))
    unknown_result = settle(writer, turn_id, unknown, execution_outcome="unknown")
    not_started_result = settle(writer, turn_id, unstarted, execution_outcome="not_started")
    terminal = finish(writer, turn_id, batch, interrupted=True)

    result = SessionReplay(store).replay(writer.session_id, SYSTEM_PROMPT)

    assert result.state.current_turn.finished == terminal.payload
    assert result.state.history[-3:-1] == (unknown_result, not_started_result)
    assert result.pending_calls == ()
    assert result.unknown_call_ids == frozenset({unknown.call_id})
    assert result.consumed_call_ids == frozenset()
    assert result.projection == ContextBuilder().project(writer.history, SYSTEM_PROMPT)
    assert "unfinished_turn" not in diagnostic_codes(result)


def test_repeat_replay_never_writes_or_executes(session, tmp_path, monkeypatch):
    store, writer = session
    turn_id = start_turn(writer)
    read = call("read")
    reply(writer, turn_id, calls=(read,))
    settle(writer, turn_id, read)
    finish(writer, turn_id, reply(writer, turn_id, consumed=(read.call_id,)))
    before = snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Replay invoked a write, tool, or model")

    original_open = Path.open

    def read_only_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            pytest.fail(f"Replay opened {path.name} for writing")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", read_only_open)
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(JsonlEventStore, "append", forbidden)
        patch.setattr(JsonlEventStore, "publish", forbidden)
        patch.setattr(SessionEventWriter, "append", forbidden)
        patch.setattr(ToolRegistry, "execute", forbidden)
        patch.setattr(LLMClient, "complete", forbidden)
        replay = SessionReplay(store)
        first = replay.replay(writer.session_id, SYSTEM_PROMPT)
        second = replay.replay(writer.session_id, SYSTEM_PROMPT)

    assert first == second
    assert snapshot(tmp_path) == before


def test_replays_copied_jsonl_without_creating_a_sidecar(session, tmp_path):
    store, writer = session
    turn_id = start_turn(writer)
    finish(writer, turn_id, reply(writer, turn_id))
    copied = tmp_path / "copied"
    target = log_path(copied, writer.session_id)
    target.parent.mkdir(parents=True)
    target.write_bytes(log_path(store.root, writer.session_id).read_bytes())
    before = snapshot(copied)

    result = SessionReplay(JsonlEventStore(copied)).replay(writer.session_id, SYSTEM_PROMPT)

    assert result.state.history == writer.history
    assert result.projection == ContextBuilder().project(writer.history, SYSTEM_PROMPT)
    assert snapshot(copied) == before
    assert not (copied / "locks").exists()


def test_retries_when_a_writer_claims_a_copied_log_during_read(session, tmp_path, monkeypatch):
    _, writer = session
    start_turn(writer)
    creation, user = writer.history
    root = tmp_path / "copied-with-race"
    path = log_path(root, writer.session_id)
    path.parent.mkdir(parents=True)
    path.write_bytes(creation.model_dump_json().encode("utf-8") + b"\n")
    reader = JsonlEventStore(root)
    concurrent_writer = JsonlEventStore(root)
    original_read = reader._read

    def read_then_append(*args, **kwargs):
        history = original_read(*args, **kwargs)
        concurrent_writer.append(user)
        return history

    replay = SessionReplay(reader)
    with monkeypatch.context() as patch:
        patch.setattr(reader, "_read", read_then_append)
        with pytest.raises(EventStoreBusyError):
            replay.replay(writer.session_id, SYSTEM_PROMPT)

    result = replay.replay(writer.session_id, SYSTEM_PROMPT)
    assert result.state.history == (creation, user)
    assert diagnostic_codes(result) == {"unfinished_turn"}


def test_rejects_missing_and_empty_logs_without_creating_a_sidecar(tmp_path):
    root = tmp_path / "logs"
    session_id = uuid4()
    replay = SessionReplay(JsonlEventStore(root))
    with pytest.raises(EventLogNotFoundError):
        replay.replay(session_id, SYSTEM_PROMPT)
    assert not root.exists()

    path = log_path(root, session_id)
    path.parent.mkdir(parents=True)
    path.touch()
    before = snapshot(root)
    with pytest.raises(EmptySessionLogError):
        replay.replay(session_id, SYSTEM_PROMPT)
    assert snapshot(root) == before
    assert not (root / "locks").exists()


@pytest.mark.parametrize("bad_record", [b"{bad}\n", b'{"incomplete":'])
def test_rejects_damaged_log_with_store_position_and_preserves_bytes(session, tmp_path, bad_record):
    store, writer = session
    isolated = tmp_path / "damaged"
    corrupt = log_path(isolated, writer.session_id)
    corrupt.parent.mkdir(parents=True)
    original = log_path(store.root, writer.session_id).read_bytes() + bad_record
    corrupt.write_bytes(original)
    before = snapshot(isolated)

    with pytest.raises(EventStoreCorruptError) as caught:
        SessionReplay(JsonlEventStore(isolated)).replay(writer.session_id, SYSTEM_PROMPT)

    assert (caught.value.session_id, caught.value.line_number, caught.value.last_valid_seq) == (
        writer.session_id, 2, 1,
    )
    assert snapshot(isolated) == before
    assert corrupt.read_bytes() == original


def test_rejects_semantic_error_even_when_log_records_are_parseable(session, tmp_path):
    store, writer = session
    turn_id = start_turn(writer)
    read = call("read")
    reply(writer, turn_id, calls=(read,))
    settle(writer, turn_id, read)
    events = writer.history
    mismatched = events[-1].model_copy(update={
        "payload": events[-1].payload.model_copy(update={"call_id": uuid4()}),
    })
    expected = (*events[:-1], mismatched)
    bad_store = JsonlEventStore(tmp_path / "inconsistent")
    path = log_path(bad_store.root, writer.session_id)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"".join(event.model_dump_json().encode("utf-8") + b"\n" for event in expected))
    before = snapshot(bad_store.root)

    with pytest.raises(HistoryError, match="next accepted call"):
        SessionReplay(bad_store).replay(writer.session_id, SYSTEM_PROMPT)

    assert snapshot(bad_store.root) == before


def test_projection_failure_keeps_reconstructed_state_and_diagnostics(session):
    class RefusingBuilder(ContextBuilder):
        def project(self, history, system_prompt):
            raise HistoryError("Cannot project this history.")

    store, writer = session
    turn_id = start_turn(writer)
    terminal = finish(writer, turn_id, reply(writer, turn_id))

    result = SessionReplay(store, builder=RefusingBuilder()).replay(writer.session_id, SYSTEM_PROMPT)

    assert result.projection is None
    assert result.state.current_turn.finished == terminal.payload
    assert result.state.history == writer.history
    assert diagnostic_codes(result) == {"projection_failed"}
