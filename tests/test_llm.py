from types import SimpleNamespace
import json
import sys

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError

from minicodeagent.agent import AgentLoop
from minicodeagent.config import Settings
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ChatMessage, ChatRequest, ModelError, ToolCall, ToolSpec
from minicodeagent.tools.builtin import register_builtin_tools
from minicodeagent.tools.registry import ToolRegistry
from minicodeagent.workspace import Workspace


class FakeSDK:
    def __init__(self, *responses):
        self.requests = []
        self.responses = iter(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.closed = False

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def sdk_call(name="read_file", arguments='{"path":"a.py"}', call_id="channel-id"):
    return SimpleNamespace(type="function", id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def sdk_reply(content=None, *, calls=(), finish="stop", reasoning=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=calls, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage)


def settings(*, thinking=False):
    environ = {
        "MINICODEAGENT_API_KEY": "unit-api-key",
        "MINICODEAGENT_BASE_URL": "https://gateway.example/compatible/v1",
        "MINICODEAGENT_MODEL": "deepseek4.1f",
        "MINICODEAGENT_TIMEOUT": "17.5",
    }
    if thinking:
        environ["MINICODEAGENT_DEEPSEEK_THINKING"] = "disabled"
    return Settings.from_env(environ)


def thinking_rejected():
    response = httpx.Response(422, request=httpx.Request("POST", "https://gateway.example"))
    return BadRequestError(
        "raw response must remain private",
        response=response,
        body={"error":{"param":"thinking", "message":"unsupported thinking configuration"}},
    )


def test_sdk_configuration_and_non_thinking_tool_feedback(monkeypatch):
    sdk = FakeSDK(sdk_reply(calls=(sdk_call(),), finish="tool_calls"), sdk_reply("done"))
    options = {}
    def openai_factory(**kwargs):
        options.update(kwargs)
        return sdk
    monkeypatch.setattr("minicodeagent.llm.OpenAI", openai_factory)
    client = LLMClient(settings(thinking=True))
    assert options == {
        "api_key": "unit-api-key", "base_url": "https://gateway.example/compatible/v1",
        "timeout": 17.5, "max_retries": 0,
    }
    user = ChatMessage(role="user", content="read a.py")
    tool = ToolSpec(name="read_file", description="Read text", parameters={"type":"object"})
    reply = client.complete(ChatRequest(messages=(user,), tools=(tool,)))
    call = reply.tool_calls[0]
    assert call.provider_call_id == "channel-id"
    assert call.arguments_raw == '{"path":"a.py"}'
    assert client.complete(ChatRequest(messages=(
        user, ChatMessage(role="assistant", content="", tool_calls=(call,)),
        ChatMessage(role="tool", content="line", call_id=call.call_id),
    ), tools=(tool,))).content == "done"
    first, second = sdk.requests
    assert first["tools"] == [{"type":"function","function":{
        "name":"read_file","description":"Read text","parameters":{"type":"object"},
    }}]
    for request in sdk.requests:
        assert request["model"] == "deepseek4.1f"
        assert request["extra_body"] == {"thinking":{"type":"disabled"}}
    assert second["messages"][-2]["tool_calls"][0] == {
        "id":call.wire_id, "type":"function", "function":{"name":"read_file","arguments":call.arguments_raw},
    }
    assert second["messages"][-1] == {"role":"tool","content":"line","tool_call_id":call.wire_id}
    client.close()
    assert sdk.closed


def test_generic_request_has_no_deepseek_fields_and_parses_usage():
    sdk = FakeSDK(sdk_reply("answer", usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
    reply = LLMClient(settings(), sdk_client=sdk).complete(ChatRequest(messages=()))
    assert reply.usage.total_tokens == 5
    assert sdk.requests == [{"model":"deepseek4.1f","messages":[]}]


def summary_request():
    return ChatRequest(messages=(
        ChatMessage(role="system", content="Produce a JSON coding handoff from evidence."),
        ChatMessage(role="user", content='{"goal": "fix a.py", "source_seq": 2}'),
    ))


def test_summary_sdk_payload_matches_preflight_and_stays_separate_from_main_request():
    body = '{"task_goal": "fix a.py"}'
    usage = SimpleNamespace(prompt_tokens=40, completion_tokens=12, total_tokens=52)
    sdk = FakeSDK(sdk_reply(body, usage=usage), sdk_reply("main reply"))
    client = LLMClient(settings(thinking=True), sdk_client=sdk)
    request = summary_request()
    preflight = client.summary_payload(request, max_output_tokens=64)
    assert not sdk.requests
    reply = client.summarize(request, max_output_tokens=64)
    assert reply.content == body and reply.usage.output_tokens == 12
    assert sdk.requests == [preflight]
    assert preflight == {
        "model": "deepseek4.1f",
        "messages": [
            {"role": "system", "content": request.messages[0].content},
            {"role": "user", "content": request.messages[1].content},
        ],
        "tool_choice": "none",
        "max_tokens": 64,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    main = ChatRequest(messages=(request.messages[1],), tools=(
        ToolSpec(name="read_file", description="Read text", parameters={"type": "object"}),
    ))
    assert client.complete(main).content == "main reply"
    assert sdk.requests[1] == client.request_payload(main)
    assert "max_tokens" not in sdk.requests[1] and "tool_choice" not in sdk.requests[1]


@pytest.mark.parametrize("invalid", ["tools", "tool_metadata", "roles", "output_budget"])
def test_summary_rejects_invalid_request_without_calling_sdk(invalid):
    sdk = FakeSDK()
    client = LLMClient(settings(), sdk_client=sdk)
    request = summary_request()
    reserve = 64
    if invalid == "tools":
        request = ChatRequest(messages=request.messages, tools=(
            ToolSpec(name="read_file", description="Read text", parameters={"type": "object"}),
        ))
    elif invalid == "tool_metadata":
        request = ChatRequest(messages=(request.messages[0], ChatMessage(
            role="user",
            content=request.messages[1].content,
            tool_calls=(ToolCall(
                name="read_file", provider_call_id="unexpected-id", arguments_raw='{"path":"a.py"}',
            ),),
        )))
    elif invalid == "roles":
        request = ChatRequest(messages=(request.messages[1], request.messages[0]))
    else:
        reserve = 0
    with pytest.raises(ModelError) as error:
        client.summarize(request, max_output_tokens=reserve)
    assert error.value.kind == "protocol_error"
    assert not sdk.requests


@pytest.mark.parametrize("response", [
    sdk_reply("partial", finish="length"),
    sdk_reply("blocked", finish="content_filter"),
    sdk_reply("  "),
    sdk_reply("text", calls=(sdk_call(),)),
    sdk_reply("text", finish="tool_calls"),
])
def test_summary_rejects_incomplete_empty_or_tool_reply_without_retry(response):
    sdk = FakeSDK(response)
    with pytest.raises(ModelError) as error:
        LLMClient(settings(), sdk_client=sdk).summarize(summary_request(), max_output_tokens=64)
    assert error.value.kind == "protocol_error"
    assert len(sdk.requests) == 1
    assert sdk.requests[0]["tool_choice"] == "none"
    assert "tools" not in sdk.requests[0]
    assert "extra_body" not in sdk.requests[0]


def test_summary_transport_timeout_does_not_retry():
    sdk = FakeSDK(APITimeoutError(request=httpx.Request("POST", "https://gateway.example")))
    with pytest.raises(ModelError) as error:
        LLMClient(settings(), sdk_client=sdk).summarize(summary_request(), max_output_tokens=64)
    assert error.value.kind == "timeout"
    assert len(sdk.requests) == 1


def test_malformed_arguments_preserved_and_projected_as_empty_object():
    sdk = FakeSDK(sdk_reply(calls=(sdk_call(arguments="bad-json"),), finish="tool_calls"), sdk_reply("ok"))
    client = LLMClient(settings(), sdk_client=sdk)
    call = client.complete(ChatRequest(messages=())).tool_calls[0]
    assert call.arguments_raw == "bad-json"
    client.complete(ChatRequest(messages=(
        ChatMessage(role="assistant", content="", tool_calls=(call,)),
        ChatMessage(role="tool", content="Invalid arguments", call_id=call.call_id),
    )))
    assert sdk.requests[-1]["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_thinking_parameter_rejection_is_specific_and_not_retried_by_client():
    sdk = FakeSDK(thinking_rejected())
    with pytest.raises(ModelError, match=r"thinking\.type=disabled") as exc:
        LLMClient(settings(thinking=True), sdk_client=sdk).complete(ChatRequest(messages=()))
    assert exc.value.kind == "compatibility_error" and not exc.value.retryable
    assert len(sdk.requests) == 1


@pytest.mark.parametrize(("response", "kind"), [
    (SimpleNamespace(choices=[], usage=None), "protocol_error"),
    (sdk_reply(calls=(sdk_call(call_id=""),), finish="tool_calls"), "protocol_error"),
    (sdk_reply("answer", reasoning="server returned thinking"), "compatibility_error"),
])
def test_invalid_or_thinking_response_does_not_fallback(response, kind):
    sdk = FakeSDK(response)
    with pytest.raises(ModelError) as exc:
        LLMClient(settings(thinking=True), sdk_client=sdk).complete(ChatRequest(messages=()))
    assert exc.value.kind == kind and not exc.value.retryable
    assert len(sdk.requests) == 1


def test_truncation_drops_even_incomplete_call():
    sdk = FakeSDK(sdk_reply("partial", calls=(SimpleNamespace(),), finish="length"))
    reply = LLMClient(settings(), sdk_client=sdk).complete(ChatRequest(messages=()))
    assert reply.content == "partial" and reply.finish_reason == "length" and not reply.tool_calls


@pytest.mark.parametrize(("status", "kind", "retryable"), [
    (401, "authentication", False),
    (408, "timeout", True),
    (429, "rate_limit", True),
    (502, "server_error", True),
])
def test_http_status_has_stable_classification(status, kind, retryable):
    response = httpx.Response(status, request=httpx.Request("POST", "https://gateway.example"))
    sdk = FakeSDK(APIStatusError("private diagnostic", response=response, body=None))
    with pytest.raises(ModelError) as exc:
        LLMClient(settings(), sdk_client=sdk).complete(ChatRequest(messages=()))
    assert exc.value.kind == kind and exc.value.retryable is retryable
    assert "private diagnostic" not in str(exc.value)


@pytest.mark.parametrize(("error_type", "kind"), [
    (APITimeoutError, "timeout"),
    (APIConnectionError, "connection"),
])
def test_sdk_transport_errors_are_classified_without_sdk_retry(error_type, kind):
    sdk = FakeSDK(error_type(request=httpx.Request("POST", "https://gateway.example")))
    with pytest.raises(ModelError) as exc:
        LLMClient(settings(), sdk_client=sdk).complete(ChatRequest(messages=()))
    assert exc.value.kind == kind and exc.value.retryable
    assert len(sdk.requests) == 1


@pytest.mark.parametrize("second_id", [None, "first"], ids=["missing-id", "duplicate-id"])
def test_invalid_sdk_batch_does_not_dispatch_any_write(tmp_path, second_id):
    first = sdk_call("write_file", '{"path":"first.txt","content":"one"}', call_id="first")
    second = sdk_call("write_file", '{"path":"second.txt","content":"two"}', call_id=second_id)
    if second_id is None:
        del second.id
    sdk = FakeSDK(sdk_reply(calls=(first, second), finish="tool_calls"))
    configured = settings()
    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=lambda argv, cwd: False)
    writer = SessionEventWriter()
    agent = AgentLoop(
        LLMClient(configured, sdk_client=sdk), registry, Workspace(tmp_path),
        writer, ContextBuilder(), configured.limits,
    )

    outcome = agent.run_turn("Apply the requested writes.")

    assert outcome.status == "failed"
    assert outcome.reason == "protocol_error"
    assert len(sdk.requests) == 1
    assert not list(tmp_path.iterdir())
    assert [event.type for event in writer.history] == ["user_message", "turn_finished"]


@pytest.mark.parametrize(("later_error", "reason"), [
    (thinking_rejected(), "compatibility_error"),
    (SimpleNamespace(choices=[], usage=None), "protocol_error"),
])
def test_second_request_errors_do_not_change_model_mode_or_repeat_command(tmp_path, later_error, reason):
    argv = [sys.executable, "-c", "print('command feedback from the workspace')"]
    first = sdk_reply(calls=(sdk_call("run_command", json.dumps({"argv":argv})),), finish="tool_calls")
    sdk = FakeSDK(first, later_error)
    configured = settings(thinking=True)
    approvals = []

    def approve(command, cwd):
        approvals.append((command, cwd))
        return True

    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=approve)
    writer = SessionEventWriter()
    agent = AgentLoop(
        LLMClient(configured, sdk_client=sdk), registry, Workspace(tmp_path),
        writer, ContextBuilder(), configured.limits,
    )
    assert agent.run_turn("Run a harmless command.").reason == reason
    assert approvals == [(tuple(argv), tmp_path.resolve())]
    assert len(sdk.requests) == 2
    for sent in sdk.requests:
        assert sent["extra_body"] == {"thinking":{"type":"disabled"}}
        assert sent["model"] == "deepseek4.1f"
    assistant, result = sdk.requests[-1]["messages"][-2:]
    assert assistant["tool_calls"][0]["id"] == result["tool_call_id"]
    assert "command feedback from the workspace" in result["content"]
    assert [event.type for event in writer.history] == [
        "user_message", "assistant_message", "tool_result", "turn_finished",
    ]
