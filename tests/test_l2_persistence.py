"""L2 eligibility and read-only restoration from persisted tool evidence."""

import json
from uuid import uuid4

import pytest

from minicodeagent.context.builder import ContextBuilder, l1_replacements, l2_replacements, projected_event_seqs
from minicodeagent.context.events import (
    AssistantMessagePayload, CompactionCommittedPayload, HistoryError, ModelRequestStartedPayload,
    TextReplacement, ToolResultPayload, TurnFinishedPayload, UserMessagePayload,
)
from minicodeagent.context.replay import SessionReplay, SessionReducer
from minicodeagent.context.store import JsonlEventSink, JsonlEventStore
from minicodeagent.context.writer import EventWriteError, SessionEventWriter
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionService
from test_session_resume_fork import run_session_process


def _start(writer, content):
    turn_id = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content=content))
    return turn_id


def _reply(writer, turn_id, attempt, *, calls=(), consumed=()):
    request_id = uuid4()
    writer.append("model_request_started", ModelRequestStartedPayload(
        turn_id=turn_id, request_id=request_id, attempt=attempt, trigger="initial" if attempt == 1 else "after_tools",
    ))
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id,
        request_id=request_id,
        content="",
        finish_reason="tool_calls" if calls else "length",
        tool_calls=calls,
        consumed_call_ids=consumed,
    ))


def _batch(writer, outcomes=("", ""), *, consumed=(0, 1), recent=False):
    turn_id = _start(writer, "\n\nCurrent work.\n\n")
    calls = tuple(
        ToolCall(provider_call_id=f"command-{index}", name="run_command", arguments_raw='{"argv":["python","-m","pytest"]}')
        for index, _ in enumerate(outcomes)
    )
    _reply(writer, turn_id, 1, calls=calls)
    stdout = (
        "pytest session starts\n"
        + "Checking fixture while tests are running.\n" * 40
        + "FAILED tests/test_api.py::test_update - AssertionError: expected status 200\n"
        + "Checking fixture while tests are running.\n" * 40
        + "1 failed, 4 passed in 0.03s\n"
    )
    results = []
    for call, outcome in zip(calls, outcomes, strict=True):
        details = {
            "exit_code": 1, "stdout": stdout, "stderr": "", "stdout_truncated": False,
            "stderr_truncated": False, "capture_incomplete": False,
        }
        if outcome:
            details["execution_outcome"] = outcome
        results.append(writer.append("tool_result", ToolResultPayload(
            turn_id=turn_id, call_id=call.call_id, status="error", details=details,
            output=f"Exit code: 1\nstdout:\n{stdout}\nstderr:\n",
        )))
    if consumed:
        _reply(writer, turn_id, 2, consumed=tuple(calls[index].call_id for index in consumed))
    if not recent:
        writer.append("turn_finished", TurnFinishedPayload(turn_id=turn_id, status="failed", reason="timeout"))
    return tuple(results)


@pytest.fixture
def tool_history(tmp_path):
    store = JsonlEventStore(tmp_path / "data")
    service = SessionService(store)
    project = tmp_path / "project"
    project.mkdir()
    with service.create(project) as handle:
        session_id = handle.session_id
    writer = SessionEventWriter(session_id=session_id, sink=JsonlEventSink(store, session_id))
    eligible = _batch(writer)
    unconsumed = _batch(writer, consumed=())
    sibling_unconsumed = _batch(writer, consumed=(0,))
    unknown = _batch(writer, ("unknown", ""))
    not_started = _batch(writer, ("not_started", ""))
    latest = _batch(writer, recent=True)
    blocked = (*unconsumed, *sibling_unconsumed, *unknown, *not_started, *latest)
    return store, service, writer, eligible, blocked


def _payload(history, replacements, levels=("L2",)):
    return CompactionCommittedPayload(
        base_seq=history[-1].seq, levels=levels, before_tokens=1200, after_tokens=1000, replacements=replacements,
    )


def test_l2_uses_consumption_and_protects_entire_batches(tool_history):
    store, _, writer, eligible, blocked = tool_history
    original = writer.history
    path = store.root / "sessions" / f"{writer.session_id}.jsonl"
    before = path.read_bytes()
    builder = ContextBuilder()
    baseline = builder.project(original, "System, AGENTS, runtime")
    changes = l2_replacements(original)
    assert tuple(item.target_seq for item in changes) == tuple(event.seq for event in eligible)
    projected = builder.project(original, "System, AGENTS, runtime", replacements=changes, levels=("L2",))
    assert projected.messages[0] == baseline.messages[0]
    assert projected.consumed_call_ids == baseline.consumed_call_ids
    assert tuple(m for m in projected.messages if m.role in ("user", "assistant")) == tuple(
        m for m in baseline.messages if m.role in ("user", "assistant")
    )
    by_call_id = {m.call_id: m for m in projected.messages if m.role == "tool"}
    assert all(by_call_id[event.payload.call_id].content == event.payload.output for event in blocked)
    assert tuple(by_call_id[event.payload.call_id].content for event in eligible) == tuple(
        item.content for item in changes
    )
    consumed = SessionReducer().replay(original).consumed_call_ids
    assert eligible[0].payload.call_id in consumed
    assert blocked[2].payload.call_id in consumed and blocked[3].payload.call_id not in consumed
    assert blocked[4].payload.call_id in consumed and blocked[4].payload.details["execution_outcome"] == "unknown"
    assert set(projected.consumed_call_ids) > consumed
    assert path.read_bytes() == before


def test_l2_commit_replays_across_process_and_fork_without_reextracting(tool_history, tmp_path):
    store, service, writer, eligible, _ = tool_history
    history = writer.history
    path = store.root / "sessions" / f"{writer.session_id}.jsonl"
    before = path.read_bytes()
    changes = l2_replacements(history)
    assert changes
    preview = ContextBuilder().project(history, "System", replacements=changes, levels=("L2",))
    commit = writer.append("compaction_committed", _payload(history, changes))
    committed = path.read_bytes()
    assert committed.startswith(before)
    assert writer.history[:-1] == history
    assert l2_replacements(writer.history) == ()
    in_process = SessionReplay(store).replay(writer.session_id, "System")
    assert in_process.projection == preview
    assert in_process.state.projection_replacements == changes
    assert in_process.state.consumed_call_ids == SessionReducer().replay(history).consumed_call_ids

    script = """
import json
from pathlib import Path
import sys
from uuid import UUID

from minicodeagent.context.budget import TokenEstimator
from minicodeagent.context.compression import l2
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.llm import LLMClient

def forbidden(*args, **kwargs):
    raise AssertionError("Replay must not extract, estimate, execute a model, or append.")

l2.extract_tool_result = forbidden
TokenEstimator.estimate = forbidden
LLMClient.complete = forbidden
SessionEventWriter.append = forbidden
replay = SessionReplay(JsonlEventStore(Path(sys.argv[1]))).replay(UUID(sys.argv[2]), "System")
print(replay.projection.model_dump_json())
"""
    assert json.loads(run_session_process(script, store.root, writer.session_id, cwd=tmp_path)) == preview.model_dump(
        mode="json",
    )
    assert path.read_bytes() == committed
    with service.fork(writer.session_id) as child:
        assert child.replay_result.projection.messages[1:] == preview.messages[1:]
        fork_commit = child.writer.history[-1]
        assert fork_commit.payload == commit.payload
        assert fork_commit.seq == commit.seq and fork_commit.session_id != commit.session_id
        assert child.state.history[eligible[0].seq - 1].payload == eligible[0].payload
    assert path.read_bytes() == committed


@pytest.mark.parametrize("kind", ["fabricated", "wrong_ref", "protected", "wrong_layer", "duplicate"])
def test_l2_rejects_invalid_preview_and_commit(tool_history, kind):
    store, _, writer, eligible, blocked = tool_history
    history = writer.history
    good, *_ = l2_replacements(history)
    path = store.root / "sessions" / f"{writer.session_id}.jsonl"
    before = path.read_bytes()
    levels = ("L1",) if kind == "wrong_layer" else ("L2",)
    if kind == "wrong_ref":
        replacements = (good.model_copy(update={"target_seq": eligible[1].seq}),)
    elif kind == "protected":
        replacements = (good.model_copy(update={"target_seq": blocked[0].seq}),)
    elif kind == "duplicate":
        replacements = (good, good)
    elif kind == "fabricated":
        replacements = (good.model_copy(update={"content": good.content + "\nInvented succeeded test."}),)
    else:
        replacements = (good,)
    with pytest.raises(HistoryError):
        ContextBuilder().project(history, "System", replacements=replacements, levels=levels)
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", _payload(history, replacements, levels=levels))
    assert path.read_bytes() == before


def test_l2_cannot_prepare_or_commit_with_pending_batch(tool_history):
    _, _, writer, _, _ = tool_history
    prior_history = writer.history
    changes = l2_replacements(prior_history)
    turn_id = prior_history[-1].payload.turn_id
    call = ToolCall(provider_call_id="pending", name="run_command", arguments_raw='{"argv":["python","-m","pytest"]}')
    _reply(writer, turn_id, 3, calls=(call,))
    history = writer.history
    with pytest.raises(HistoryError, match="unsettled"):
        l2_replacements(history)
    with pytest.raises(EventWriteError, match="compaction_committed"):
        writer.append("compaction_committed", _payload(history, changes))
    assert writer.history == history


def test_l1_l2_commit_applies_each_layer_and_preserves_source_order(tool_history):
    store, _, writer, _, _ = tool_history
    history = writer.history
    l1_changes = l1_replacements(history)
    l2_changes = l2_replacements(history)
    assert l1_changes and l2_changes
    changes = (*l1_changes, *l2_changes)
    expected = ContextBuilder().project(history, "System", replacements=changes, levels=("L1", "L2"))
    writer.append("compaction_committed", _payload(history, changes, levels=("L1", "L2")))
    replay = SessionReplay(store).replay(writer.session_id, "System")
    assert replay.projection == expected
    assert projected_event_seqs(replay.state.history) == projected_event_seqs(history)
    assert replay.state.history[:-1] == history
