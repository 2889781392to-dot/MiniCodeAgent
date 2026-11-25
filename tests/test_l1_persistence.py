"""Validate the L1 overlay against immutable history on commit and replay."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from minicodeagent.context.budget import TokenEstimator
from minicodeagent.context.builder import ContextBuilder, l1_replacements, projected_event_seqs
from minicodeagent.context.compression import l1
from minicodeagent.context.events import (
    AssistantMessagePayload,
    CompactionCommittedPayload,
    HistoryError,
    ModelRequestStartedPayload,
    SessionEvent,
    TextReplacement,
    ToolResultPayload,
    TurnFinishedPayload,
    UserMessagePayload,
)
from minicodeagent.context.replay import SessionReducer, SessionReplay
from minicodeagent.context.store import JsonlEventSink, JsonlEventStore
from minicodeagent.context.writer import EventWriteError, SessionEventWriter
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionService


def start_turn(writer, content):
    turn = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn, content=content))
    return turn


def start_request(writer, turn, attempt):
    request = ModelRequestStartedPayload(
        turn_id=turn, request_id=uuid4(), attempt=attempt, trigger="initial" if attempt == 1 else "after_tools",
    )
    writer.append("model_request_started", request)
    return request


def batch(writer, turn, attempt, outputs):
    request = start_request(writer, turn, attempt)
    calls = tuple(
        ToolCall(provider_call_id=f"call-{attempt}-{index}", name="read_file", arguments_raw='{"path":"a.py"}')
        for index, _ in enumerate(outputs)
    )
    message = writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn, request_id=request.request_id, content="\nCalling tools.\n\n",
        tool_calls=calls, finish_reason="tool_calls",
    ))
    results = [
        writer.append("tool_result", ToolResultPayload(
            turn_id=turn, call_id=call.call_id, status=status, output=output,
            details={"execution_outcome": "unknown"} if status in {"interrupted", "timeout"} else {},
        ))
        for call, (status, output) in zip(calls, outputs, strict=True)
    ]
    return message, results


@pytest.fixture
def session(tmp_path):
    store = JsonlEventStore(tmp_path / "data")
    service = SessionService(store)
    project = tmp_path / "project"
    project.mkdir()
    with service.create(project) as handle:
        session_id = handle.session_id
    writer = SessionEventWriter(session_id=session_id, sink=JsonlEventSink(store, session_id))
    old = start_turn(writer, "\r\n\nOld task\r\n\r\n\n")
    old_batch, old_results = batch(writer, old, 1, [
        ("interrupted", "\nOld unknown output\n\n"), ("timeout", "\nOld timeout output\n\n"),
    ])
    writer.append("turn_finished", TurnFinishedPayload(turn_id=old, status="failed", reason="timeout"))
    current = start_turn(writer, "\nLatest target intact\n\n")
    batch(writer, current, 1, [("ok", "first")])
    batch(writer, current, 2, [("ok", "second"), ("ok", "third")])
    path = store.root / "sessions" / f"{session_id}.jsonl"
    return store, service, writer, current, old_batch, old_results, path


def compaction(history, *changes):
    return CompactionCommittedPayload(
        base_seq=len(history), before_tokens=120, after_tokens=100, replacements=changes,
    )


def test_l1_preview_commit_replay_and_fork_keep_original_prefix(session, monkeypatch):
    store, service, writer, _, _, old_results, path = session
    before = path.read_bytes()
    history = writer.history
    builder = ContextBuilder()
    full = builder.project(history, "system")
    assert full.messages[1].content == "\r\n\nOld task\r\n\r\n\n"
    replacements = l1_replacements(history)
    assert replacements == (TextReplacement(target_seq=2, content="Old task\r\n"),)
    preview = builder.project(history, "system", replacements=replacements)
    assert preview.messages[1].content == "Old task\r\n"
    assert preview.messages[2:] == full.messages[2:]
    assert preview.consumed_call_ids == full.consumed_call_ids
    assert projected_event_seqs(history)[0] == 2
    assert path.read_bytes() == before

    record = writer.append("compaction_committed", compaction(history, *replacements))
    committed = path.read_bytes()
    assert committed.startswith(before) and writer.history[:-1] == history
    assert record.seq == len(history) + 1 and len(committed.splitlines()) == len(history) + 1

    def forbidden(*args, **kwargs):
        pytest.fail("Pure replay may apply stored overlays only.")

    monkeypatch.setattr(TokenEstimator, "estimate", forbidden)
    monkeypatch.setattr(l1, "trim_display_edges", forbidden)
    replay = SessionReplay(store).replay(writer.session_id, "system")
    assert replay.projection == preview
    assert replay.state.history[:-1] == history
    assert replay.state.projection_replacements == replacements
    assert replay.state.consumed_call_ids == frozenset()
    assert replay.unknown_call_ids == {event.payload.call_id for event in old_results}
    assert path.read_bytes() == committed
    with service.fork(writer.session_id) as child:
        assert child.state.projection_replacements == replacements
        assert child.state.history[-1].payload == record.payload
        assert child.state.history[-1].session_id != record.session_id
    assert path.read_bytes() == committed


@pytest.mark.parametrize("target", ["latest_user", "batch_head", "tool_result", "stale", "fabricated", "duplicate"])
def test_invalid_l1_changes_fail_both_preview_and_append(session, target):
    _, _, writer, current, old_batch, old_results, path = session
    history = writer.history
    original = path.read_bytes()
    good, = l1_replacements(history)
    seq = {
        "latest_user": next(e.seq for e in history if e.type == "user_message" and e.payload.turn_id == current),
        "batch_head": old_batch.seq,
        "tool_result": old_results[0].seq,
        "stale": len(history) + 1,
        "fabricated": good.target_seq,
    }.get(target)
    content = {
        "latest_user": "Latest target intact\n",
        "batch_head": "Calling tools.\n",
        "tool_result": "Old unknown output\n",
    }.get(target, "fabricated")
    changes = (good, good) if target == "duplicate" else (TextReplacement(target_seq=seq, content=content),)
    with pytest.raises(HistoryError):
        ContextBuilder().project(history, "system", replacements=changes)
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", compaction(history, *changes))
    assert path.read_bytes() == original


def test_l1_rejects_stale_base_and_duplicate_commits(session):
    _, _, writer, _, _, _, path = session
    history = writer.history
    payload = compaction(history, *l1_replacements(history))
    before = path.read_bytes()
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", payload.model_copy(update={"base_seq": payload.base_seq - 1}))
    assert path.read_bytes() == before
    original = writer.append("compaction_committed", payload)
    committed = path.read_bytes()
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", payload.model_copy(update={"base_seq": original.seq}))
    assert path.read_bytes() == committed
    assert l1_replacements(writer.history) == ()


def test_last_two_complete_blocks_can_cross_turn_boundaries():
    writer = SessionEventWriter()
    old = start_turn(writer, "\nOriginal question\n\n")
    request = start_request(writer, old, 1)
    final = writer.append("assistant_message", AssistantMessagePayload(
        turn_id=old, request_id=request.request_id, content="\nCurrent answer\n\n", finish_reason="stop",
    ))
    writer.append("turn_finished", TurnFinishedPayload(
        turn_id=old, status="completed", reason="model_stop", assistant_seq=final.seq,
    ))
    start_turn(writer, "Next goal")
    history = writer.history
    assert l1_replacements(history) == (TextReplacement(target_seq=1, content="Original question\n"),)
    with pytest.raises(HistoryError, match="protected"):
        ContextBuilder().project(history, "system", replacements=(
            TextReplacement(target_seq=final.seq, content="Current answer\n"),
        ))


def test_persistent_corrupt_overlay_is_rejected_without_repair(session):
    store, _, writer, _, old_batch, _, path = session
    history = writer.history
    # Store can parse and read the envelope. The shared reducer owns semantic validation.
    invalid = SessionEvent(
        session_id=writer.session_id, event_id=uuid4(), seq=len(history) + 1, ts=datetime.now(UTC),
        type="compaction_committed",
        payload=compaction(history, TextReplacement(target_seq=old_batch.seq, content="Calling tools.\n")),
    )
    store.append(invalid)
    before = path.read_bytes()
    assert tuple(store.iter_events(writer.session_id))[-1] == invalid
    with pytest.raises(HistoryError, match="protected"):
        SessionReplay(store).replay(writer.session_id, "system")
    assert path.read_bytes() == before


def test_unsettled_batch_blocks_l1_projection_across_reopen(session):
    store, _, writer, current, _, _, path = session
    request = start_request(writer, current, 3)
    call = ToolCall(provider_call_id="pending", name="read_file", arguments_raw="{}")
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=current, request_id=request.request_id, content="", finish_reason="tool_calls", tool_calls=(call,),
    ))
    before = path.read_bytes()
    history = writer.history
    with pytest.raises(HistoryError, match="unsettled"):
        l1_replacements(history)
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", compaction(history, TextReplacement(target_seq=2, content="Old task\r\n")))
    replay = SessionReplay(store).replay(writer.session_id, "system")
    assert replay.projection is None and replay.pending_calls == (call,)
    assert replay.state == SessionReducer().replay(history)
    assert path.read_bytes() == before
