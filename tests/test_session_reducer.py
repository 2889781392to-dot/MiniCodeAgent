import builtins
from datetime import UTC, datetime
from pathlib import Path
import socket
import subprocess
from uuid import uuid4

import pytest

from minicodeagent.context.events import (
    AssistantMessagePayload,
    HistoryError,
    SessionCreatedPayload,
    SessionEvent,
    ToolResultPayload,
    TurnFinishedPayload,
    UserMessagePayload,
)
from minicodeagent.context.replay import SessionReducer, SessionState, TurnState
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.protocol import ToolCall


TIMESTAMP = datetime(2026, 11, 20, tzinfo=UTC)


@pytest.fixture
def creation(tmp_path):
    return SessionEvent(
        session_id=uuid4(), event_id=uuid4(), seq=1, ts=TIMESTAMP,
        type="session_created",
        payload=SessionCreatedPayload(title="Persisted project", workspace=str(tmp_path / "project")),
    )


def event_after(previous, kind, payload):
    return SessionEvent(
        session_id=previous.session_id, event_id=uuid4(), seq=previous.seq + 1,
        ts=TIMESTAMP, type=kind, payload=payload,
    )


def user_after(previous):
    return event_after(
        previous, "user_message", UserMessagePayload(turn_id=uuid4(), content="Inspect the file"),
    )


def assistant_after(previous, turn_id, *, calls=(), consumed=(), rejected=()):
    return event_after(
        previous, "assistant_message",
        AssistantMessagePayload(
            turn_id=turn_id, request_id=uuid4(), content="Here is the result",
            finish_reason="tool_calls" if calls or rejected else "stop",
            tool_calls=calls, consumed_call_ids=consumed, rejected_tool_calls=rejected,
        ),
    )


def tool_result_after(previous, turn_id, call, *, interrupted=False):
    return event_after(
        previous, "tool_result",
        ToolResultPayload(
            turn_id=turn_id, call_id=call.call_id, output="Recorded tool result",
            status="interrupted" if interrupted else "ok",
            details={"execution_outcome": "unknown"} if interrupted else {},
        ),
    )


def finish_after(previous, turn_id, *, assistant_seq=None, interrupted=False):
    return event_after(
        previous, "turn_finished",
        TurnFinishedPayload(
            turn_id=turn_id, assistant_seq=assistant_seq,
            status="interrupted" if interrupted else "completed",
            reason="interrupted" if interrupted else "model_stop",
        ),
    )


def call(provider_id):
    return ToolCall(provider_call_id=provider_id, name="read_file", arguments_raw='{"path":"app.py"}')


def test_empty_history_and_initial_log_position():
    state = SessionReducer().replay(iter(()))
    assert state == SessionState()
    assert state.session_id is None and state.metadata is None and state.current_turn is None
    assert state.history == () and state.consumed_call_ids == frozenset()
    assert (state.last_seq, state.next_seq) == (0, 1)


def test_creation_initializes_metadata_and_log_position(creation):
    state = SessionState()
    assert SessionReducer().apply(state, creation) is None
    assert (state.session_id, state.metadata, state.history) == (
        creation.session_id, creation.payload, (creation,),
    )
    assert state.current_turn is None
    assert (state.last_seq, state.next_seq) == (1, 2)


def test_stop_reply_keeps_turn_open_until_recorded_terminal(creation):
    reducer = SessionReducer()
    user = user_after(creation)
    reply = assistant_after(user, user.payload.turn_id)
    state = reducer.replay((creation, user, reply))
    assert state.current_turn == TurnState(
        turn_id=user.payload.turn_id, user_seq=user.seq, tool_batches=0,
        pending_calls=(), latest_reply=reply, finished=None,
    )
    finished = finish_after(reply, user.payload.turn_id, assistant_seq=reply.seq)
    reducer.apply(state, finished)
    assert state.current_turn.latest_reply == reply
    assert state.current_turn.finished == finished.payload
    assert state.last_seq == finished.seq


def test_multiple_tools_stay_pending_in_declaration_order_until_settled(creation):
    reducer = SessionReducer()
    user = user_after(creation)
    turn_id = user.payload.turn_id
    first, second = call("first"), call("second")
    batch = assistant_after(user, turn_id, calls=(first, second))
    state = reducer.replay((creation, user, batch))
    assert state.current_turn.tool_batches == 1
    assert state.current_turn.pending_calls == (first, second)

    with pytest.raises(HistoryError):
        reducer.apply(state, tool_result_after(batch, turn_id, second))
    with pytest.raises(HistoryError):
        reducer.apply(state, finish_after(batch, turn_id, assistant_seq=batch.seq))
    with pytest.raises(HistoryError):
        reducer.apply(state, assistant_after(batch, turn_id))

    first_result = tool_result_after(batch, turn_id, first)
    reducer.apply(state, first_result)
    assert state.current_turn.pending_calls == (second,)
    interrupted_result = tool_result_after(first_result, turn_id, second, interrupted=True)
    reducer.apply(state, interrupted_result)
    assert state.current_turn.pending_calls == ()
    assert state.current_turn.finished is None
    assert state.history[-1] == interrupted_result
    assert state.consumed_call_ids == frozenset()

    finished = finish_after(interrupted_result, turn_id, assistant_seq=batch.seq, interrupted=True)
    reducer.apply(state, finished)
    assert state.current_turn.finished.status == "interrupted"
    assert state.history[-2].payload.details == {"execution_outcome": "unknown"}


def test_consumption_is_recorded_evidence_across_multiple_turns(creation):
    reducer = SessionReducer()
    first_user = user_after(creation)
    first_turn = first_user.payload.turn_id
    accepted, rejected = call("accepted"), call("rejected")
    batch = assistant_after(first_user, first_turn, calls=(accepted,))
    result = tool_result_after(batch, first_turn, accepted)
    first_terminal = finish_after(result, first_turn, assistant_seq=batch.seq, interrupted=True)
    second_user = user_after(first_terminal)
    second_turn = second_user.payload.turn_id
    state = reducer.replay((creation, first_user, batch, result, first_terminal, second_user))
    assert state.current_turn.turn_id == second_turn
    assert state.current_turn.user_seq == second_user.seq
    assert state.current_turn.tool_batches == 0 and state.current_turn.finished is None
    assert state.consumed_call_ids == frozenset()

    invalid = assistant_after(second_user, second_turn, consumed=(uuid4(),))
    with pytest.raises(HistoryError):
        reducer.apply(state, invalid)
    reply = assistant_after(
        second_user, second_turn, consumed=(accepted.call_id,), rejected=(rejected,),
    )
    reducer.apply(state, reply)
    assert state.consumed_call_ids == frozenset({accepted.call_id})
    assert state.current_turn.pending_calls == ()
    assert state.current_turn.tool_batches == 0


@pytest.mark.parametrize("duplicated_identity", ["local", "provider"])
def test_rejected_duplicate_batch_leaves_incremental_state_unchanged(creation, duplicated_identity):
    reducer = SessionReducer()
    user = user_after(creation)
    valid = call("same")
    duplicate = (
        valid.model_copy(update={"provider_call_id": "other"})
        if duplicated_identity == "local" else call("same")
    )
    bad = assistant_after(user, user.payload.turn_id, calls=(valid, duplicate))
    state = reducer.replay((creation, user))
    before = reducer.replay(state.history)
    with pytest.raises(HistoryError):
        reducer.apply(state, bad)
    assert state == before
    fixed = bad.model_copy(update={"payload": bad.payload.model_copy(update={"tool_calls": (valid,)})})
    reducer.apply(state, fixed)
    assert state.current_turn.pending_calls == (valid,)


def test_rejects_reused_local_call_id_after_settlement(creation):
    reducer = SessionReducer()
    user = user_after(creation)
    turn_id = user.payload.turn_id
    first = call("wire")
    called = assistant_after(user, turn_id, calls=(first,))
    result = tool_result_after(called, turn_id, first)
    state = reducer.replay((creation, user, called, result))
    duplicate = assistant_after(result, turn_id, calls=(first,))
    with pytest.raises(HistoryError):
        reducer.apply(state, duplicate)
    new_call_same_provider = call("wire")
    reducer.apply(state, assistant_after(result, turn_id, calls=(new_call_same_provider,)))
    assert state.current_turn.pending_calls == (new_call_same_provider,)
    assert state.current_turn.tool_batches == 2


@pytest.mark.parametrize(
    "field", ["missing_header", "duplicate_header", "seq_gap", "seq_repeat", "event_id", "session_id"],
)
def test_rejects_invalid_log_order_and_envelope(creation, field):
    reducer = SessionReducer()
    user = user_after(creation)
    if field == "missing_header":
        events = (user.model_copy(update={"seq": 1}),)
    elif field == "duplicate_header":
        events = (creation, creation.model_copy(update={"seq": 2, "event_id": uuid4()}))
    else:
        invalid = {
            "seq_gap": {"seq": 3},
            "seq_repeat": {"seq": 1},
            "event_id": {"event_id": creation.event_id},
            "session_id": {"session_id": uuid4()},
        }[field]
        events = (creation, user.model_copy(update=invalid))
    with pytest.raises(HistoryError):
        reducer.replay(events)


def test_rejects_duplicate_terminal_and_cross_turn_assistant_reference(creation):
    reducer = SessionReducer()
    first_user = user_after(creation)
    first_turn = first_user.payload.turn_id
    first_reply = assistant_after(first_user, first_turn)
    first_terminal = finish_after(first_reply, first_turn, assistant_seq=first_reply.seq)
    state = reducer.replay((creation, first_user, first_reply, first_terminal))

    with pytest.raises(HistoryError):
        reducer.apply(state, finish_after(first_terminal, first_turn, assistant_seq=first_reply.seq))
    assert state.current_turn.finished == first_terminal.payload
    second_user = user_after(first_terminal)
    reducer.apply(state, second_user)
    with pytest.raises(HistoryError):
        reducer.apply(state, finish_after(
            second_user, second_user.payload.turn_id, assistant_seq=first_reply.seq,
        ))
    assert state.current_turn.finished is None


def test_rejects_overlapping_turns_and_reused_turn_and_request_ids(creation):
    reducer = SessionReducer()
    user = user_after(creation)
    reply = assistant_after(user, user.payload.turn_id)
    state = reducer.replay((creation, user, reply))
    with pytest.raises(HistoryError):
        reducer.apply(state, user_after(reply))
    finished = finish_after(reply, user.payload.turn_id, assistant_seq=reply.seq)
    reducer.apply(state, finished)
    with pytest.raises(HistoryError):
        reducer.apply(state, event_after(finished, "user_message", user.payload))
    next_user = user_after(finished)
    reducer.apply(state, next_user)
    reused_request = reply.payload.model_copy(update={"turn_id": next_user.payload.turn_id})
    with pytest.raises(HistoryError):
        reducer.apply(state, event_after(next_user, "assistant_message", reused_request))
    assert state.current_turn.latest_reply is None


def test_jsonl_readback_and_incremental_state_match_pure_reduction(tmp_path, creation):
    user = user_after(creation)
    turn_id = user.payload.turn_id
    accepted = call("wire")
    batch = assistant_after(user, turn_id, calls=(accepted,))
    result = tool_result_after(batch, turn_id, accepted)
    reply = assistant_after(result, turn_id, consumed=(accepted.call_id,))
    finished = finish_after(reply, turn_id, assistant_seq=reply.seq)
    events = (creation, user, batch, result, reply, finished)
    store = JsonlEventStore(tmp_path / "logs")
    store.publish(creation.session_id, events)
    path = tmp_path / "logs" / "sessions" / f"{creation.session_id}.jsonl"
    original = path.read_bytes()

    reducer = SessionReducer()
    from_disk = reducer.replay(store.iter_events(creation.session_id))
    incremental = SessionState()
    for event in events:
        reducer.apply(incremental, event)
    assert from_disk == incremental == reducer.replay(events) == reducer.replay(iter(events))
    assert from_disk.history == events
    assert path.read_bytes() == original


def test_reducer_does_not_touch_external_services(creation, monkeypatch):
    user = user_after(creation)
    accepted = call("wire")
    batch = assistant_after(user, user.payload.turn_id, calls=(accepted,))
    events = (creation, user, batch)
    encoded = tuple(event.model_dump_json() for event in events)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected filesystem, network, or subprocess access during reduction")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", unexpected)
        patch.setattr(Path, "open", unexpected)
        patch.setattr(Path, "stat", unexpected)
        patch.setattr(subprocess, "Popen", unexpected)
        patch.setattr(socket, "create_connection", unexpected)
        state = SessionReducer().replay(events)

    assert state.current_turn.pending_calls == (accepted,)
    assert tuple(event.model_dump_json() for event in events) == encoded
