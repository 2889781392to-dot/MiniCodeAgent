from uuid import UUID

import pytest

from minicodeagent.agent import AgentLoop
from minicodeagent.config import LoopLimits
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.writer import MemoryEventSink, SessionEventWriter
from minicodeagent.protocol import ModelError, ModelReply, ProtocolModel, ToolCall, ToolResult
from minicodeagent.tools.registry import ToolDefinition, ToolRegistry
from minicodeagent.workspace import Workspace


class EmptyArgs(ProtocolModel):
    pass


class FakeModel:
    def __init__(self, *responses):
        self.requests = []
        self.responses = iter(responses)

    def complete(self, request):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def call(provider_id, *, name="record", raw="{}"):
    return ToolCall(provider_call_id=provider_id, name=name, arguments_raw=raw)


def batch(*calls):
    return ModelReply(content="", finish_reason="tool_calls", tool_calls=calls)


def make_agent(tmp_path, model, handler, *, writer=None, limits=None):
    registry = ToolRegistry()
    registry.register(ToolDefinition("record", "Test-only tool", EmptyArgs, handler))
    return AgentLoop(
        model, registry, Workspace(tmp_path), writer or SessionEventWriter(),
        ContextBuilder(), limits or LoopLimits(),
    )


def test_unknown_and_invalid_calls_settle_before_next_request(tmp_path):
    first = call("first")
    missing = call("missing", name="missing")
    invalid = call("invalid", raw="{")
    last = call("last")
    model = FakeModel(batch(first, missing, invalid, last), ModelReply(content="Done", finish_reason="stop"))
    executed = []

    def handler(args, workspace, call_id):
        current_ids = [event.payload.call_id for event in agent.writer.history if event.type == "tool_result"]
        assert current_ids == ([] if call_id == first.call_id else [first.call_id, missing.call_id, invalid.call_id])
        executed.append(call_id)
        return ToolResult(call_id=call_id, status="ok", output="success")

    agent = make_agent(tmp_path, model, handler)
    outcome = agent.run_turn("test the batch")
    assert outcome.status == "completed"
    assert executed == [first.call_id, last.call_id]
    assert len(model.requests) == 2
    messages = model.requests[-1].messages
    assert messages[-5].tool_calls == (first, missing, invalid, last)
    assert tuple(message.call_id for message in messages[-4:]) == (
        first.call_id, missing.call_id, invalid.call_id, last.call_id,
    )
    assert "Unknown tool" in messages[-3].content
    assert "valid JSON object" in messages[-2].content
    history = agent.writer.history
    assert history[-2].payload.consumed_call_ids == tuple(message.call_id for message in messages[-4:])
    assert [event.type for event in history].count("turn_finished") == 1


@pytest.mark.parametrize(("reply", "limits", "expected"), [
    (batch(call("dup"), call("dup")), LoopLimits(), "protocol_error"),
    (batch(), LoopLimits(), "protocol_error"),
    (ModelReply(finish_reason="length", tool_calls=(call("x"),)), LoopLimits(), "output_length"),
    (ModelReply(finish_reason="content_filter", tool_calls=(call("x"),)), LoopLimits(), "content_filter"),
    (batch(call("a"), call("b")), LoopLimits(max_calls_per_batch=1), "tool_batch_size_limit"),
    (batch(call("a")), LoopLimits(max_tool_batches=0), "tool_batch_limit"),
    (batch(call("a")), LoopLimits(max_provider_calls=1), "provider_call_limit"),
])
def test_bad_or_unaffordable_batch_has_no_side_effects(tmp_path, reply, limits, expected):
    model = FakeModel(reply)
    agent = make_agent(tmp_path, model, lambda *args: pytest.fail("unexpected tool execution"), limits=limits)
    result = agent.run_turn("unsafe batch")
    assert result.reason == expected
    assert len(model.requests) == 1
    assert [event.type for event in agent.writer.history].count("turn_finished") == 1
    assert not any(event.type == "tool_result" for event in agent.writer.history)
    assert not any(
        event.payload.tool_calls for event in agent.writer.history if event.type == "assistant_message"
    )


@pytest.mark.parametrize("kind", ["connection", "timeout", "rate_limit", "server_error"])
def test_single_transient_retry_is_applied_to_the_same_request(tmp_path, kind):
    first = call("first")
    model = FakeModel(
        batch(first), ModelError(kind, "retry once"), ModelReply(content="Done", finish_reason="stop"),
    )
    ran = []

    def handler(args, workspace, call_id):
        ran.append(call_id)
        return ToolResult(call_id=call_id, status="ok", output="done")

    agent = make_agent(tmp_path, model, handler)
    assert agent.run_turn("test retry").status == "completed"
    assert ran == [first.call_id]
    assert model.requests[-2] is model.requests[-1]


def test_transient_retry_budget_is_shared_across_rounds(tmp_path):
    first = call("first")
    error = ModelError("server_error", "transient")
    model = FakeModel(error, batch(first), error)
    ran = []

    def handler(args, workspace, call_id):
        ran.append(call_id)
        return ToolResult(call_id=call_id, status="ok", output="done")

    agent = make_agent(tmp_path, model, handler)
    outcome = agent.run_turn("test retry exhaustion")
    assert (outcome.status, outcome.reason) == ("failed", "server_error")
    assert ran == [first.call_id]
    assert len(model.requests) == 3


@pytest.mark.parametrize("fail_at", ["user_message", "assistant_message", "tool_result", "turn_finished"])
def test_writer_failure_prevents_further_dispatch(tmp_path, fail_at):
    class BadSink(MemoryEventSink):
        def append(self, event):
            if event.type == fail_at:
                raise OSError("write failed")
            super().append(event)

    first, second = call("first"), call("second")
    model = FakeModel(batch(first, second), ModelReply(content="Done", finish_reason="stop"))
    ran = []
    def handler(args, workspace, call_id):
        ran.append(call_id)
        return ToolResult(call_id=call_id, status="ok", output="done")

    agent = make_agent(tmp_path, model, handler, writer=SessionEventWriter(sink=BadSink()))
    outcome = agent.run_turn("test failed writer")
    assert (outcome.status, outcome.reason) == ("failed", "writer_error")
    assert outcome.content == f"Could not append {fail_at}."
    executed_count = {"user_message": 0, "assistant_message": 0, "tool_result": 1, "turn_finished": 2}
    assert ran == [first.call_id, second.call_id][:executed_count[fail_at]]
    assert len(model.requests) == {"user_message": 0, "assistant_message": 1, "tool_result": 1, "turn_finished": 2}[fail_at]
    assert "turn_finished" not in [event.type for event in agent.writer.history]


def test_interrupt_settles_started_and_unstarted_calls(tmp_path):
    first, second = call("first"), call("second")
    model = FakeModel(batch(first, second))
    def handler(args, workspace, call_id: UUID):
        if call_id == first.call_id:
            raise KeyboardInterrupt
        pytest.fail("unexpected dispatch after interruption")

    agent = make_agent(tmp_path, model, handler)
    assert agent.run_turn("interrupt").status == "interrupted"
    results = [event.payload for event in agent.writer.history if event.type == "tool_result"]
    assert [result.call_id for result in results] == [first.call_id, second.call_id]
    assert [result.details["execution_outcome"] for result in results] == ["unknown", "not_started"]
    assert agent.writer.history[-1].type == "turn_finished"
    assert len(model.requests) == 1


@pytest.mark.parametrize(("limits", "reason"), [
    (LoopLimits(max_provider_calls=2), "provider_call_limit"),
    (LoopLimits(max_tool_batches=1), "tool_batch_limit"),
])
def test_limit_rejects_followup_batch_without_losing_first_result(tmp_path, limits, reason):
    first, second = call("first"), call("second")
    model = FakeModel(batch(first), batch(second))
    ran = []

    def handler(args, workspace, call_id):
        ran.append(call_id)
        return ToolResult(call_id=call_id, status="ok", output="first result")

    agent = make_agent(tmp_path, model, handler, limits=limits)
    outcome = agent.run_turn("first and second tool calls")

    assert (outcome.status, outcome.reason) == ("limited", reason)
    assert ran == [first.call_id]
    assert len(model.requests) == 2
    assert model.requests[0].messages[-1].content == "first and second tool calls"
    assert model.requests[1].messages[-2].tool_calls == (first,)
    assert model.requests[1].messages[-1].call_id == first.call_id
    assert model.requests[1].messages[-1].content == "first result"
    rejected, finished = agent.writer.history[-2:]
    assert rejected.payload.tool_calls == ()
    assert rejected.payload.rejected_tool_calls == (second,)
    assert rejected.payload.consumed_call_ids == (first.call_id,)
    assert finished.payload.assistant_seq == rejected.seq


@pytest.mark.parametrize(("kind", "limits"), [
    ("authentication", LoopLimits()),
    ("protocol_error", LoopLimits()),
    ("connection", LoopLimits(max_provider_calls=1)),
])
def test_model_failure_without_retry_closes_once(tmp_path, kind, limits):
    model = FakeModel(ModelError(kind, "request failed"))
    agent = make_agent(tmp_path, model, lambda *args: pytest.fail("unexpected dispatch"), limits=limits)
    outcome = agent.run_turn("model failure")

    assert (outcome.status, outcome.reason) == ("failed", kind)
    assert len(model.requests) == 1
    assert [event.type for event in agent.writer.history] == ["user_message", "turn_finished"]
    assert agent.writer.history[-1].payload.reason == kind
    assert agent.writer.history[-1].payload.assistant_seq is None


def test_reused_local_call_id_is_rejected_after_first_result(tmp_path):
    first = call("first")
    reused = ToolCall(
        provider_call_id="another provider ID", name=first.name, arguments_raw="{}", call_id=first.call_id,
    )
    model = FakeModel(batch(first), batch(reused))
    ran = []

    def handler(args, workspace, call_id):
        ran.append(call_id)
        return ToolResult(call_id=call_id, status="ok", output="done")

    agent = make_agent(tmp_path, model, handler)
    assert agent.run_turn("do not repeat the same call").reason == "protocol_error"
    assert ran == [first.call_id]
    assert [event.type for event in agent.writer.history] == [
        "user_message", "assistant_message", "tool_result", "turn_finished",
    ]


def test_unvalidated_typed_reply_with_missing_identity_rejects_entire_batch(tmp_path):
    first = call("valid")
    missing_id = call("before update").model_copy(update={"provider_call_id": ""})
    model = FakeModel(batch(first, missing_id), ModelReply(content="Unexpected follow-up", finish_reason="stop"))
    ran = []

    def handler(args, workspace, call_id):
        ran.append(call_id)
        return ToolResult(call_id=call_id, status="ok", output="done")

    agent = make_agent(tmp_path, model, handler)
    outcome = agent.run_turn("reject the whole invalid batch")
    assert ran == []
    assert (outcome.status, outcome.reason) == ("failed", "protocol_error")
    assert [event.type for event in agent.writer.history] == ["user_message", "turn_finished"]


def test_registry_remains_fixed_through_retry_and_dispatch(tmp_path):
    class CheckedModel(FakeModel):
        def complete(self, request):
            with pytest.raises(RuntimeError, match="in use"):
                agent.registry.register(definition)
            return super().complete(request)

    def handler(args, workspace, call_id):
        with pytest.raises(RuntimeError, match="in use"):
            agent.registry.register(definition)
        return ToolResult(call_id=call_id, status="ok", output="unchanged binding")

    definition = ToolDefinition("record", "Test-only tool", EmptyArgs, handler)
    model = CheckedModel(ModelError("timeout", "retry"), batch(call("accepted")),
                         ModelReply(content="Done", finish_reason="stop"))
    agent = make_agent(tmp_path, model, handler)
    assert agent.run_turn("fixed binding").status == "completed"
    assert len(model.requests) == 3
    assert all(request.tools == model.requests[0].tools for request in model.requests)
    assert model.requests[-1].messages[-1].content == "unchanged binding"
    with pytest.raises(ValueError, match="already registered"):
        agent.registry.register(definition)


@pytest.mark.parametrize("interrupt_at", ["before_commit", "after_commit", "subscriber"])
def test_interrupt_during_terminal_append_preserves_one_committed_outcome(tmp_path, interrupt_at):
    class InterruptingSink(MemoryEventSink):
        interrupted = False

        def append(self, event):
            if event.type == "turn_finished" and not self.interrupted:
                self.interrupted = True
                if interrupt_at == "after_commit":
                    super().append(event)
                raise KeyboardInterrupt
            super().append(event)

    interrupted = False

    def interrupt_subscriber(event):
        nonlocal interrupted
        if event.type == "turn_finished" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    model = FakeModel(ModelReply(content="Done", finish_reason="stop"),
                      ModelReply(content="Next turn", finish_reason="stop"))
    writer = (SessionEventWriter(on_event=interrupt_subscriber) if interrupt_at == "subscriber"
              else SessionEventWriter(sink=InterruptingSink()))
    agent = make_agent(tmp_path, model, lambda *args: pytest.fail("unexpected dispatch"),
                       writer=writer)

    try:
        outcome = agent.run_turn("finish the first turn")
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped during the terminal append")
    assert outcome.status == ("interrupted" if interrupt_at == "before_commit" else "completed")
    assert [event.type for event in agent.writer.history] == ["user_message", "assistant_message", "turn_finished"]
    finished = agent.writer.history[-1]
    assert finished.payload.status == outcome.status
    assert finished.payload.assistant_seq == agent.writer.history[-2].seq
    assert agent.run_turn("start the second turn").status == "completed"
