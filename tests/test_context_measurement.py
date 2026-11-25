"""Measure the same attempts that actually cross the model boundary."""

import json
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from openai.types.completion_usage import CompletionUsage

from minicodeagent import cli, llm
from minicodeagent.agent import AgentLoop
from minicodeagent.config import LoopLimits
from minicodeagent.context.budget import BudgetPolicy, TokenEstimator
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.events import SessionCreatedPayload
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventSink, JsonlEventStore
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ModelError, ModelReply, ToolCall
from minicodeagent.tools.builtin import register_builtin_tools
from minicodeagent.tools.registry import ToolRegistry
from minicodeagent.workspace import Workspace
from test_llm import FakeSDK, sdk_call, sdk_reply, settings
from test_python_environment_cli import capture_model


class CapturingEstimator(TokenEstimator):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def estimate(self, request, *, system_parts=None, payload=None):
        self.inputs.append((request, system_parts, payload))
        return super().estimate(request, system_parts=system_parts, payload=payload)


def setup_agent(tmp_path, client, estimator, **options):
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("value = 42\n", encoding="utf-8")
    (project / "AGENTS.md").write_text(
        "保持测试原样。\n[End project background]\nDon't parse markers.", encoding="utf-8", newline="\n",
    )
    store = JsonlEventStore(tmp_path / "data")
    session_id = uuid4()
    writer = SessionEventWriter(session_id=session_id, sink=JsonlEventSink(store, session_id))
    writer.append("session_created", SessionCreatedPayload(title="Test", workspace=str(project.resolve())))
    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=lambda *_args: pytest.fail("No commands expected."))
    agent = AgentLoop(
        client, registry, Workspace(project), writer, ContextBuilder(), LoopLimits(),
        track_requests=True, token_estimator=estimator, system_prompt="Inspect before editing.",
        environment_prompt=lambda: "[Current run environment observation]\n{}\n[End current run environment observation]",
        **options,
    )
    return agent, store, session_id


def test_fake_llm_observes_exact_requests_tool_rounds_and_retry(tmp_path):
    estimator = CapturingEstimator()
    calls = [ToolCall(provider_call_id=f"read-{index}", name="read_file", arguments_raw='{"path":"a.py"}')
             for index in range(2)]
    requests = []

    class Client:
        def complete(self, request):
            requests.append(request)
            index = len(requests) - 1
            assert estimator.inputs[index][0] is request
            assert agent.writer.history[-1].type == "model_request_started"
            assert agent.writer.history[-1].payload.attempt == index + 1
            if index == 1:
                raise ModelError("timeout", "fake timeout")
            if index in (0, 2):
                return ModelReply(tool_calls=(calls[int(index == 2)],), finish_reason="tool_calls")
            return ModelReply(content="done", finish_reason="stop")

    snapshots = []
    agent, store, session_id = setup_agent(
        tmp_path, Client(), estimator, on_measurement=snapshots.append,
        budget_policy=BudgetPolicy(model_context_window=100, output_reserve=5),
    )
    assert agent.run_turn("Inspect a.py").status == "completed"
    events = tuple(store.iter_events(session_id))
    starts = [e.payload for e in events if e.type == "model_request_started"]
    replies = [e.payload for e in events if e.type == "assistant_message"]
    assert agent.measurements == tuple(snapshots)
    assert len(requests) == len(snapshots) == len(starts) == 4
    assert [(s.request_id, s.turn_id, s.attempt) for s in snapshots] == [
        (e.request_id, e.turn_id, e.attempt) for e in starts
    ]
    assert [s.trigger for s in snapshots] == ["initial", "after_tools", "retry_transient", "after_tools"]
    assert [s.failure_kind for s in snapshots] == [None, "timeout", None, None]
    assert [s.request_id for s in snapshots if s.failure_kind is None] == [e.request_id for e in replies]
    assert requests[1] is requests[2] and snapshots[1].estimate == snapshots[2].estimate
    assert requests[1].messages[-1].call_id == calls[0].call_id
    assert requests[3].messages[-1].call_id == calls[1].call_id
    assert all(s.usage_status == "unavailable" for s in snapshots)
    assert all(s.dispatch_boundary == "model_client" and s.at_or_above_high_watermark for s in snapshots)
    assert all(s.at_or_above_input_capacity for s in snapshots)
    assert [r.role for r in requests[0].messages] == ["system", "user"]
    for request, parts, _ in estimator.inputs:
        assert parts.text == request.messages[0].content
        assert "[End project background]\nDon't parse markers." in parts.project_memory
        assert parts.runtime_context.startswith("\n\n[Current run environment observation]")
    print(json.dumps({
        "request_id": str(snapshots[0].request_id), "attempt": snapshots[0].attempt,
        "messages": [m.model_dump(mode="json", exclude_defaults=True) for m in requests[0].messages],
        "tool_names": [tool.name for tool in requests[0].tools],
        "tokens": snapshots[0].estimate.estimated_input_tokens,
        "attempt_tokens": [s.estimate.estimated_input_tokens for s in snapshots],
    }, ensure_ascii=True))


def test_sdk_payload_and_prompt_usage_are_correlated_without_event_schema_change(tmp_path):
    estimator = CapturingEstimator()
    actual = CompletionUsage(
        prompt_tokens=180, completion_tokens=12, total_tokens=192,
        prompt_tokens_details={"cached_tokens": 64, "audio_tokens": 0, "cache_write_tokens": 9},
    )
    sdk = FakeSDK(sdk_reply(calls=(sdk_call(arguments="broken["),), finish="tool_calls", usage=actual),
                  sdk_reply("done"))
    agent, store, session_id = setup_agent(tmp_path, LLMClient(settings(thinking=True), sdk_client=sdk), estimator)
    assert agent.run_turn("Inspect a.py").status == "completed"
    assert len(sdk.requests) == len(estimator.inputs) == len(agent.measurements) == 2
    for sent, (_, _, measured) in zip(sdk.requests, estimator.inputs):
        assert sent == measured and sent["messages"] is measured["messages"] and sent["tools"] is measured["tools"]
    assert sdk.requests[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert sdk.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sdk.requests[1]["model"] == "deepseek4.1f"
    replies = [e.payload for e in store.iter_events(session_id) if e.type == "assistant_message"]
    first, second = agent.measurements
    assert [s.request_id for s in agent.measurements] == [r.request_id for r in replies]
    assert first.dispatch_boundary == second.dispatch_boundary == "sdk_create"
    assert first.usage_status == "available" and second.usage_status == "unavailable"
    assert (first.actual_usage.input_tokens, first.actual_usage.output_tokens, first.actual_usage.total_tokens) == (180, 12, 192)
    assert first.prompt_tokens_details.model_dump() == {"cached_tokens": 64, "audio_tokens": 0, "cache_write_tokens": 9}
    assert first.prompt_token_delta == 180 - first.estimate.estimated_input_tokens
    assert second.actual_usage is second.prompt_token_delta is second.prompt_tokens_details is None
    assert first.actual_usage == replies[0].usage
    assert replies[0].usage.model_dump() == {"input_tokens": 180, "output_tokens": 12, "total_tokens": 192}
    assert "prompt_tokens_details" not in replies[0].model_dump()
    print(json.dumps({"request_id": str(first.request_id), "estimate": first.estimate.estimated_input_tokens,
                      "usage": first.actual_usage.model_dump(), "details": first.prompt_tokens_details.model_dump(),
                      "delta": first.prompt_token_delta, "later_usage": second.usage_status}))


def test_failure_before_sdk_dispatch_has_no_measurement(tmp_path, monkeypatch):
    def reject_before_dispatch(*_args, **_kwargs):
        raise ModelError("protocol_error", "Could not render the request.")

    sdk = FakeSDK()
    estimator = CapturingEstimator()
    agent, store, session_id = setup_agent(tmp_path, LLMClient(settings(), sdk_client=sdk), estimator)
    monkeypatch.setattr(llm, "encode_chat_payload", reject_before_dispatch)
    assert agent.run_turn("Inspect a.py").reason == "protocol_error"
    assert not agent.measurements and not estimator.inputs and not sdk.requests
    assert [e.type for e in store.iter_events(session_id)] == [
        "session_created", "user_message", "model_request_started", "turn_finished",
    ]


def test_optional_sdk_details_and_missing_usage_keep_distinct_states(tmp_path):
    usage = SimpleNamespace(
        prompt_tokens=12, completion_tokens=0, total_tokens=12,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, audio_tokens=-1, cache_write_tokens="bad"),
    )
    reply_without_usage = sdk_reply("No usage")
    del reply_without_usage.usage
    sdk = FakeSDK(sdk_reply("A real zero", usage=usage), reply_without_usage)
    agent, _, _ = setup_agent(tmp_path, LLMClient(settings(), sdk_client=sdk), TokenEstimator())
    assert agent.run_turn("First task").status == "completed"
    first, = agent.measurements
    assert first.prompt_tokens_details.model_dump(exclude_none=True) == {"cached_tokens": 0}
    assert first.usage_status == "available"
    assert agent.run_turn("Second task").status == "completed"
    second, = agent.measurements
    assert second.usage_status == "unavailable" and second.prompt_tokens_details is None
    assert second.request_id != first.request_id and second.turn_id != first.turn_id


def test_failed_estimator_is_observational_and_replay_never_counts(tmp_path, monkeypatch):
    sdk = FakeSDK(sdk_reply("done"))
    client = LLMClient(settings(), sdk_client=sdk)
    snapshots = []
    agent, store, session_id = setup_agent(tmp_path, client, TokenEstimator(), on_measurement=snapshots.append)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Sensitive implementation detail")

    monkeypatch.setattr(TokenEstimator, "estimate", unavailable)
    result = agent.run_turn("Inspect a.py")
    assert result.reason == "model_stop" and len(sdk.requests) == 1
    snapshot, = snapshots
    assert snapshot.estimate.status == "unavailable"
    assert "Sensitive" not in snapshot.estimate.diagnostic
    assert snapshot.estimate.estimated_input_tokens is None
    path = store.root / "sessions" / f"{session_id}.jsonl"
    before = path.read_bytes()

    def forbidden(*_args, **_kwargs):
        pytest.fail("Replay must not estimate tokens, send a model request, or write events.")

    monkeypatch.setattr(TokenEstimator, "estimate", forbidden)
    monkeypatch.setattr(LLMClient, "complete", forbidden)
    monkeypatch.setattr(SessionEventWriter, "append", forbidden)
    replay = SessionReplay(store).replay(session_id, "Replay")
    assert replay.state.current_turn.finished.reason == "model_stop"
    assert replay.projection.messages[-1].content == "done"
    assert len(replay.state.history) == 5 and path.read_bytes() == before and agent.measurements == (snapshot,)


def test_cli_measurement_output_is_opt_in_and_matches_model_request(tmp_path, monkeypatch, capsys):
    requests = capture_model(monkeypatch)
    options = ["--workspace", str(tmp_path), "--python", sys.executable, "--message", "fixed user"]
    monkeypatch.setattr(cli, "confirm_command", lambda *_args: False)
    assert cli.main(options) == 0
    assert "[context measurement]" not in capsys.readouterr().err
    assert cli.main([*options, "--measure-context"]) == 0
    line, = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[context measurement] ")]
    data = json.loads(line.removeprefix("[context measurement] "))
    assert requests[0] == requests[1]
    assert data["attempt"] == 1 and data["usage_status"] == "unavailable"
    assert data["estimate"]["message_count"] == len(requests[1].messages)
    assert data["estimate"]["status"] == "estimated"
    assert data["estimate"]["breakdown"]["history_tokens"] > 0
    assert data["estimate"]["estimated_input_tokens"] > 0 and data["at_or_above_high_watermark"] is None
    assert data["at_or_above_input_capacity"] is None
    assert "fixed user" not in line
