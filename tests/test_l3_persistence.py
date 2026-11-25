"""Exercise source-linked extractive checkpoints on the existing JSONL path."""

from datetime import UTC, datetime
import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from minicodeagent.context.builder import (
    ContextBuilder, l1_replacements, l2_replacements, l3_bounds, l3_checkpoint, projected_event_seqs,
)
from minicodeagent.context.events import (
    AssistantMessagePayload, CompactionCommittedPayload, ExtractiveCheckpoint, HistoryError,
    ModelRequestStartedPayload, SessionEvent, TextReplacement, TurnFinishedPayload, UserMessagePayload,
)
from minicodeagent.context.replay import SessionReducer, SessionReplay
from minicodeagent.context.store import JsonlEventSink, JsonlEventStore
from minicodeagent.context.writer import EventWriteError, SessionEventWriter
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionService
from test_l2_persistence import _batch
from test_session_resume_fork import run_session_process


def _start(writer, content):
    turn_id = uuid4()
    return writer.append("user_message", UserMessagePayload(turn_id=turn_id, content=content))


def _plain(writer, goal, reply):
    user = _start(writer, goal)
    turn_id, request_id = user.payload.turn_id, uuid4()
    writer.append("model_request_started", ModelRequestStartedPayload(
        turn_id=turn_id, request_id=request_id, attempt=1, trigger="initial",
    ))
    answer = writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=request_id, finish_reason="stop", content=reply,
    ))
    writer.append("turn_finished", TurnFinishedPayload(
        turn_id=turn_id, status="completed", reason="model_stop", assistant_seq=answer.seq,
    ))
    return user, answer


def _commit(history, checkpoint, *, replacements=(), levels=("L3",)):
    return CompactionCommittedPayload(
        base_seq=history[-1].seq, before_tokens=10000, after_tokens=7500,
        levels=levels, replacements=replacements, checkpoint=checkpoint,
    )


@pytest.fixture
def journal(tmp_path):
    store = JsonlEventStore(tmp_path / "data")
    service = SessionService(store)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with service.create(workspace) as handle:
        session_id = handle.session_id
    writer = SessionEventWriter(session_id=session_id, sink=JsonlEventSink(store, session_id))
    path = store.root / "sessions" / f"{session_id}.jsonl"
    return store, service, writer, path


def test_l3_boundary_replay_fork_and_l2_overlay_retirement(journal, tmp_path):
    store, service, writer, path = journal
    results = _batch(writer)
    recent_goal, recent_answer = _plain(writer, "Check path routing.", "Retain this verification in the latest tail.")
    latest = _start(writer, "Latest goal, exact wording.")
    initial = writer.history
    originals = l2_replacements(initial)
    assert len(originals) == 2
    writer.append("compaction_committed", CompactionCommittedPayload(
        base_seq=initial[-1].seq, before_tokens=7000, after_tokens=6000, levels=("L2",), replacements=originals,
    ))
    history, before = writer.history, path.read_bytes()
    bounds = l3_bounds(history)
    assert bounds.start_seq == 1
    assert bounds.covered_through_seq == recent_goal.seq
    assert bounds.covered_events == recent_goal.seq
    assert bounds.protected_from_seq == recent_answer.seq
    proposed = l3_checkpoint(history)
    assert proposed == l3_checkpoint(history)
    assert proposed.covered_through_seq == recent_goal.seq
    assert proposed.kind == "extractive"
    assert any(
        f"[{event.seq}]" in entry and str(event.payload.call_id) in entry
        for entry in proposed.summary for event in results
    )
    system = "System, project memory, runtime."
    preview = ContextBuilder().project(history, system, checkpoint=proposed, levels=("L3",))
    assert preview.messages[0].content == system
    assert preview.messages[1].role == "user"
    assert [m.content for m in preview.messages[2:]] == [recent_answer.payload.content, latest.payload.content]
    assert preview.consumed_call_ids == ()
    assert path.read_bytes() == before

    event = writer.append("compaction_committed", _commit(history, proposed))
    after = path.read_bytes()
    assert after.startswith(before)
    state = SessionReducer().replay(writer.history)
    assert state.history[:-1] == history
    assert state.checkpoint == proposed
    assert state.projection_replacements == ()
    assert state.consumed_call_ids == {event.payload.call_id for event in results}
    assert projected_event_seqs(writer.history) == (recent_answer.seq, latest.seq)
    assert l3_checkpoint(writer.history) is None and l2_replacements(writer.history) == ()
    assert SessionReplay(store).replay(writer.session_id, system).projection == preview

    script = """
import json
from pathlib import Path
import sys
from uuid import UUID
from minicodeagent.context.budget import TokenEstimator
from minicodeagent.context.compression import l3
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.tools.registry import ToolRegistry

def forbidden(*args, **kwargs):
    raise AssertionError("Pure Replay must use persisted facts only.")

l3.extract_summary = forbidden
TokenEstimator.estimate = forbidden
SessionEventWriter.append = forbidden
LLMClient.complete = forbidden
ToolRegistry.execute = forbidden
result = SessionReplay(JsonlEventStore(Path(sys.argv[1]))).replay(
    UUID(sys.argv[2]), "System, project memory, runtime.",
)
print(json.dumps(result.projection.model_dump(mode="json"), ensure_ascii=True))
"""
    restored = run_session_process(script, store.root, writer.session_id, cwd=tmp_path)
    assert json.loads(restored) == preview.model_dump(mode="json")
    with service.fork(writer.session_id) as child:
        assert child.state.checkpoint == proposed
        assert child.replay_result.projection.messages[1:] == preview.messages[1:]
        child_commit = child.state.history[-1]
        assert child_commit.session_id != event.session_id and child_commit.event_id != event.event_id
        assert child_commit.seq == event.seq and child_commit.payload == event.payload
    assert path.read_bytes() == after


@pytest.mark.parametrize(("outcomes", "consumed"), [
    (("", ""), (0,)), (("unknown", ""), (0, 1)), (("not_started", ""), (0, 1)),
])
def test_earliest_protected_transaction_stops_l3_prefix(journal, outcomes, consumed):
    _, _, writer, path = journal
    _plain(writer, "Fix a previous task.", "Verified earlier work.")
    blocked = _batch(writer, outcomes, consumed=consumed)
    allowed_after_blocker = _batch(writer)
    latest = _start(writer, "Keep this target.")
    history, before = writer.history, path.read_bytes()
    bounds = l3_bounds(history)
    assert bounds.protected_from_seq == blocked[0].seq - 1
    checkpoint = l3_checkpoint(history)
    assert checkpoint.covered_through_seq < bounds.protected_from_seq
    projection = ContextBuilder().project(history, "System", checkpoint=checkpoint, levels=("L3",))
    assert set(projection.consumed_call_ids) == {item.payload.call_id for item in (*blocked, *allowed_after_blocker)}
    assert projection.messages[-1].content == latest.payload.content
    bad = checkpoint.model_copy(update={"covered_through_seq": blocked[-1].seq})
    with pytest.raises(HistoryError):
        ContextBuilder().project(history, "System", checkpoint=bad, levels=("L3",))
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", _commit(history, bad))
    assert path.read_bytes() == before


def test_l3_checkpoint_extends_without_overlapping_or_repeating_inherited_evidence(journal):
    store, _, writer, path = journal
    _plain(writer, "Fix totals.py", "Verified totals.py behavior.")
    _plain(writer, "Check config.py next.", "Reviewed config.py.")
    first_goal = _start(writer, "Investigate cli.py.")
    history = writer.history
    first = l3_checkpoint(history)
    writer.append("compaction_committed", _commit(history, first))
    assert first is not None and l3_checkpoint(writer.history) is None
    turn_id, request_id = first_goal.payload.turn_id, uuid4()
    writer.append("model_request_started", ModelRequestStartedPayload(
        turn_id=turn_id, request_id=request_id, attempt=1, trigger="initial",
    ))
    answer = writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=request_id, finish_reason="stop", content="Tests confirmed cli.py behavior.",
    ))
    writer.append("turn_finished", TurnFinishedPayload(
        turn_id=turn_id, status="completed", reason="model_stop", assistant_seq=answer.seq,
    ))
    _plain(writer, "Read helpers.py.", "Retain the latest helper evidence.")
    latest = _start(writer, "Keep my new task.")
    history, before = writer.history, path.read_bytes()
    second = l3_checkpoint(history)
    assert l3_bounds(history).start_seq == first.covered_through_seq + 1
    assert second.covered_through_seq > first.covered_through_seq
    assert set(first.summary) <= set(second.summary)
    assert len(second.summary) == len(set(second.summary))
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", _commit(history, first))
    writer.append("compaction_committed", _commit(history, second))
    assert path.read_bytes().startswith(before)
    replay = SessionReplay(store).replay(writer.session_id, "System")
    assert replay.state.checkpoint == second
    assert [e.payload.checkpoint for e in replay.state.history if e.type == "compaction_committed"] == [first, second]
    assert replay.projection.messages[-1].content == latest.payload.content
    assert sum(message.content.startswith("[Extractive checkpoint") for message in replay.projection.messages) == 1
    assert sum(message.role == "system" for message in replay.projection.messages) == 1
    assert l3_checkpoint(writer.history) is None


@pytest.mark.parametrize("bad", [
    {"kind": "handoff", "covered_through_seq": 8, "summary": ["a"]},
    {"covered_through_seq": 8, "summary": ["a"]},
    {"kind": "extractive", "covered_through_seq": 8},
    {"kind": "extractive", "covered_through_seq": 0, "summary": ["a"]},
])
def test_l3_wire_schema_requires_extractive_checkpoint_fields(bad):
    with pytest.raises(ValidationError):
        ExtractiveCheckpoint.model_validate_json(json.dumps(bad))


def test_pending_and_forged_l3_sources_cannot_commit(journal):
    _, _, writer, path = journal
    batch = _batch(writer)
    _plain(writer, "Read helpers.py.", "Keep helper evidence in the tail.")
    current = _start(writer, "Check another protected call.")
    history, before = writer.history, path.read_bytes()
    checkpoint = l3_checkpoint(history)
    wrong_text = checkpoint.model_copy(update={"summary": (*checkpoint.summary, "目标 [9999]：Invented.")})
    old_call_id = next(
        str(item.payload.call_id) for item in batch
        if any(str(item.payload.call_id) in row for row in checkpoint.summary)
    )
    wrong_call = checkpoint.model_copy(update={
        "summary": tuple(item.replace(old_call_id, str(uuid4())) for item in checkpoint.summary),
    })
    assert wrong_call.summary != checkpoint.summary
    for bad in (wrong_text, wrong_call, checkpoint.model_copy(update={"covered_through_seq": batch[0].seq})):
        with pytest.raises(EventWriteError):
            writer.append("compaction_committed", _commit(history, bad))
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", _commit(
            history, checkpoint, levels=("L1", "L3"),
            replacements=(TextReplacement(target_seq=2, content="Current work.\n"),),
        ))
    assert path.read_bytes() == before
    request_id = uuid4()
    writer.append("model_request_started", ModelRequestStartedPayload(
        turn_id=current.payload.turn_id, request_id=request_id, attempt=1, trigger="initial",
    ))
    pending_batch = writer.append("assistant_message", AssistantMessagePayload(
        turn_id=current.payload.turn_id, request_id=request_id, finish_reason="tool_calls",
        content="", tool_calls=(ToolCall(provider_call_id="pending", name="run_command", arguments_raw="{}"),),
    ))
    pending = writer.history
    assert l3_bounds(pending).stop_reason == "pending_tool_calls"
    assert pending_batch.seq in l3_bounds(pending).protected_batches
    assert l3_checkpoint(pending) is None
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", _commit(pending, checkpoint))
    assert writer.history == pending


def test_combined_l1_l3_commit_preserves_eligible_tail_overlay(journal):
    store, _, writer, path = journal
    _plain(writer, "Earlier verified task.", "Keep the old evidence.")
    blocked = _batch(writer, consumed=())
    tail_goal, tail_answer = _plain(writer, "\n\nInspect helpers.py.\n\n", "Keep this latest reply.")
    _start(writer, "Keep this latest user goal.")
    history, before = writer.history, path.read_bytes()
    checkpoint = l3_checkpoint(history)
    assert checkpoint.covered_through_seq < blocked[0].seq - 1
    changes = tuple(item for item in l1_replacements(history) if item.target_seq > checkpoint.covered_through_seq)
    assert changes == (TextReplacement(target_seq=tail_goal.seq, content="Inspect helpers.py.\n"),)
    preview = ContextBuilder().project(history, "System", checkpoint=checkpoint, levels=("L1", "L3"),
                                       replacements=changes)
    writer.append("compaction_committed", _commit(history, checkpoint, levels=("L1", "L3"), replacements=changes))
    assert path.read_bytes().startswith(before)
    replay = SessionReplay(store).replay(writer.session_id, "System")
    assert replay.state.history[:-1] == history
    assert replay.state.projection_replacements == changes
    assert replay.projection == preview
    assert replay.projection.messages[-2].content == tail_answer.payload.content
    assert set(replay.projection.consumed_call_ids) == {item.payload.call_id for item in blocked}


def test_persisted_forged_checkpoint_fails_replay_without_repair(journal):
    store, _, writer, path = journal
    _plain(writer, "Fix an older bug.", "Verified earlier behavior.")
    _plain(writer, "Review the current diagnosis.", "Keep this recent report.")
    _start(writer, "Do not drop this goal.")
    history, before = writer.history, path.read_bytes()
    checkpoint = l3_checkpoint(history).model_copy(update={"summary": ("目标 [999]：No source exists.",)})
    injected = SessionEvent(
        session_id=writer.session_id, event_id=uuid4(), seq=len(history) + 1,
        ts=datetime.now(UTC), type="compaction_committed", payload=_commit(history, checkpoint),
    )
    store.append(injected)  # Bypass Writer's semantic validation to exercise a corrupt persisted record.
    corrupted = path.read_bytes()
    assert corrupted.startswith(before)
    with pytest.raises(HistoryError, match="original evidence"):
        SessionReplay(store).replay(writer.session_id, "System")
    assert path.read_bytes() == corrupted
