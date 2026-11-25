import json
import os
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from minicodeagent.agent import AgentLoop
from minicodeagent.config import LoopLimits, Settings
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.events import (
    AssistantMessagePayload, HistoryError, ModelRequestStartedPayload, SessionCreatedPayload,
    ToolResultPayload, UserMessagePayload,
)
from minicodeagent.context.replay import SessionReducer
from minicodeagent.context.store import JsonlEventSink, JsonlEventStore
from minicodeagent.context.writer import EventWriteError, SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ModelError, ModelReply, ToolCall
from minicodeagent.tools.registry import ToolRegistry
from minicodeagent.workspace import Workspace


def make_writer(tmp_path):
    writer = SessionEventWriter()
    writer.append("session_created", SessionCreatedPayload(
        title="Task", workspace=str(tmp_path.resolve()),
    ))
    return writer


def make_agent(tmp_path, writer, client, **kwargs):
    return AgentLoop(
        client, ToolRegistry(), Workspace(tmp_path), writer, ContextBuilder(),
        LoopLimits(**kwargs), track_requests=True, system_prompt="Persistent session.",
    )


def add_request(writer, turn_id, attempt, trigger, request_id=None):
    payload = ModelRequestStartedPayload(
        turn_id=turn_id, request_id=request_id or uuid4(), attempt=attempt, trigger=trigger,
    )
    writer.append("model_request_started", payload)
    return payload


def test_persistent_attempts_recorded_before_dispatch_and_reply_matches_retry(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = JsonlEventStore(tmp_path / "data")
    session_id = uuid4()
    writer = SessionEventWriter(session_id=session_id, sink=JsonlEventSink(store, session_id))
    writer.append("session_created", SessionCreatedPayload(
        title="Task", workspace=str(workspace.resolve()),
    ))
    requests = []

    class Client:
        def complete(self, request):
            start = tuple(store.iter_events(session_id))[-1].payload
            assert isinstance(start, ModelRequestStartedPayload)
            requests.append(start)
            assert request.messages[0].content == "Persistent session."
            if len(requests) == 1:
                raise ModelError("timeout", "retry")
            return ModelReply(content="answer", finish_reason="stop")

    agent = make_agent(workspace, writer, Client())
    result = agent.run_turn("task")
    assert result.status == "completed"
    assert [start.attempt for start in requests] == [1, 2]
    assert [start.trigger for start in requests] == ["initial", "retry_transient"]
    assert requests[0].request_id != requests[1].request_id
    state = SessionReducer().replay(store.iter_events(session_id))
    turn = state.current_turn
    assert (turn.model_attempts, turn.retried_transient) == (2, True)
    assert turn.latest_request == requests[-1]
    assert turn.latest_reply.payload.request_id == requests[-1].request_id


@pytest.mark.parametrize("fault", ["append", "fsync"])
def test_request_start_storage_failure_prevents_sdk_dispatch(tmp_path, monkeypatch, fault):
    store = JsonlEventStore(tmp_path / "data")
    session_id = uuid4()
    writer = SessionEventWriter(session_id=session_id, sink=JsonlEventSink(store, session_id))
    writer.append("session_created", SessionCreatedPayload(
        title="Task", workspace=str(tmp_path.resolve()),
    ))
    path = store.root / "sessions" / f"{session_id}.jsonl"
    sdk_create = Mock()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=sdk_create)))
    settings = Settings.from_env({
        "MINICODEAGENT_API_KEY": "synthetic-test-key",
        "MINICODEAGENT_MODEL": "fake-model",
    })
    client = LLMClient(settings, sdk_client=sdk)
    complete = Mock(wraps=client.complete)
    monkeypatch.setattr(client, "complete", complete)
    failed = []

    def reject_request_start(kind):
        if kind == "model_request_started":
            failed.append(kind)
            raise OSError(f"Injected {fault} failure")

    with monkeypatch.context() as patch:
        if fault == "append":
            append = store.append

            def fail_append(event):
                reject_request_start(event.type)
                append(event)

            patch.setattr(store, "append", fail_append)
        else:
            fsync = os.fsync

            def fail_fsync(fd):
                kind = json.loads(path.read_bytes().splitlines()[-1])["type"]
                reject_request_start(kind)
                fsync(fd)

            patch.setattr(os, "fsync", fail_fsync)
        agent = make_agent(tmp_path, writer, client)
        result = agent.run_turn("task")

    assert failed == ["model_request_started"]
    assert (result.status, result.reason) == ("failed", "writer_error")
    assert result.content == "Could not append model_request_started."
    complete.assert_not_called()
    sdk_create.assert_not_called()
    assert agent.measurements == ()
    expected = ["session_created", "user_message"]
    if fault == "fsync":
        # A flushed record may be visible even though its durability was not acknowledged.
        expected.append("model_request_started")
    assert [event.type for event in store.iter_events(session_id)] == expected


def test_continue_retains_turn_budget_transient_retry_and_prior_tool_settlement(tmp_path):
    writer = make_writer(tmp_path)
    turn_id = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="task"))
    add_request(writer, turn_id, 1, "initial")
    latest = add_request(writer, turn_id, 2, "retry_transient")
    call = ToolCall(provider_call_id="first", name="read_file", arguments_raw="{}")
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=latest.request_id, content="",
        finish_reason="tool_calls", tool_calls=(call,),
    ))
    writer.append("tool_result", ToolResultPayload(
        call_id=call.call_id, turn_id=turn_id, status="interrupted",
        output="Outcome unknown.", details={"execution_outcome": "unknown"},
    ))
    state = SessionReducer().replay(writer.history)
    original = writer.history
    observed = []

    class Client:
        def complete(self, request):
            observed.append(writer.history[-1].payload)
            assert request.messages[-1].call_id == call.call_id
            raise ModelError("timeout", "no retry remains")

    result = make_agent(tmp_path, writer, Client()).continue_turn(state)
    assert (result.turn_id, result.status, result.reason) == (turn_id, "failed", "timeout")
    assert [(start.attempt, start.trigger) for start in observed] == [(3, "resume")]
    assert writer.history[:len(original)] == original
    assert [event.type for event in writer.history[len(original):]] == [
        "model_request_started", "turn_finished",
    ]


def test_continue_refuses_legacy_tail_and_stale_snapshot_without_writing(tmp_path):
    writer = make_writer(tmp_path)
    turn_id = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="legacy or pre-request tail"))
    legacy = SessionReducer().replay(writer.history)

    class Client:
        def complete(self, _request):
            pytest.fail("must not send a request")

    agent = make_agent(tmp_path, writer, Client())
    before = writer.history
    with pytest.raises(HistoryError):
        agent.continue_turn(legacy)
    assert writer.history == before
    add_request(writer, turn_id, 1, "initial")
    live = writer.history
    with pytest.raises(HistoryError):
        agent.continue_turn(legacy)
    assert writer.history == live


def test_unanswered_attempt_consumes_budget_after_restart(tmp_path):
    writer = make_writer(tmp_path)
    turn_id = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="task"))
    add_request(writer, turn_id, 1, "initial")
    state = SessionReducer().replay(writer.history)

    class Client:
        def complete(self, _request):
            pytest.fail("budget already exhausted")

    result = make_agent(tmp_path, writer, Client(), max_provider_calls=1).continue_turn(state)
    assert (result.turn_id, result.status, result.reason) == (turn_id, "limited", "provider_call_limit")
    assert [event.type for event in writer.history[-2:]] == ["model_request_started", "turn_finished"]


def test_request_replay_rejects_sequence_gaps_stale_replies_and_duplicate_ids(tmp_path):
    writer = make_writer(tmp_path)
    turn_id = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="task"))
    initial = add_request(writer, turn_id, 1, "initial")
    with pytest.raises(EventWriteError):
        add_request(writer, turn_id, 3, "resume")
    with pytest.raises(EventWriteError):
        add_request(writer, turn_id, 2, "resume", request_id=initial.request_id)
    actual = add_request(writer, turn_id, 2, "resume")
    with pytest.raises(EventWriteError):
        writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn_id, request_id=initial.request_id, content="stale", finish_reason="stop",
        ))
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=actual.request_id, content="answer", finish_reason="stop",
    ))
    assert SessionReducer().replay(writer.history).current_turn.model_attempts == 2


def test_tracking_cannot_begin_after_unregistered_legacy_reply(tmp_path):
    writer = make_writer(tmp_path)
    turn_id = uuid4()
    writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="legacy"))
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=turn_id, request_id=uuid4(), content="legacy answer", finish_reason="stop",
    ))
    with pytest.raises(EventWriteError):
        add_request(writer, turn_id, 1, "initial")
    assert SessionReducer().replay(writer.history).current_turn.latest_request is None
