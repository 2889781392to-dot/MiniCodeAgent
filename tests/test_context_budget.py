"""Exercise the deterministic request measurement without a model or SDK."""

from copy import deepcopy
from dataclasses import asdict
import json
from uuid import UUID

import pytest

from minicodeagent.chat_payload import encode_chat_payload
from minicodeagent.context.budget import BudgetPolicy, ContextMeasurementSnapshot, SystemPromptParts, TokenEstimator
from minicodeagent.protocol import ChatMessage, ChatRequest, TokenUsage, ToolCall, ToolSpec


@pytest.fixture
def projected_request():
    parts = SystemPromptParts(
        "Follow project rules.", "\n\n[Project]\n测试约定", "\n\n[Runtime]\nPython 3.11",
    )
    call = ToolCall(
        provider_call_id="provider-private-id", call_id=UUID(int=42), name="read_file",
        arguments_raw='{"path": "fixture.py"}',
    )
    request = ChatRequest(messages=(
        ChatMessage(role="system", content=parts.text),
        ChatMessage(role="user", content="Inspect fixture.py.\nPreserve \"examples\"."),
        ChatMessage(role="assistant", tool_calls=(call,), content="Reading the file."),
        ChatMessage(role="tool", call_id=call.call_id, content="print('模型')\n"),
    ), tools=(ToolSpec(
        name="read_file", description="Return text from the workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    ),))
    return parts, request


def test_text_estimator_uses_utf8_bytes_and_rounds_up():
    estimator = TokenEstimator()
    assert [estimator.estimate_text(text) for text in ("", "abcdefgh", "你们", "🌱")] == [0, 3, 2, 2]
    with pytest.raises(ValueError, match="text"):
        estimator.estimate_text(12)


def test_request_components_partition_its_single_estimate(projected_request):
    parts, request = projected_request
    estimator = TokenEstimator()
    payload = encode_chat_payload(request, model="synthetic-model", deepseek_thinking="disabled")
    original_payload = deepcopy(payload)
    estimate = estimator.estimate(request, system_parts=parts, payload=payload)
    assert payload == original_payload
    assert estimate == estimator.estimate(request, system_parts=parts, payload=deepcopy(payload))
    assert estimate.status == "estimated" and estimate.diagnostic is None
    assert estimate.message_count == 4 and estimate.tokenizer_id and estimate.tokenizer_version
    body = {key: value for key, value in payload.items() if key != "extra_body"} | payload["extra_body"]
    compact_bytes = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert estimate.estimated_input_tokens == (compact_bytes + 2) // 3 + 3 * (estimate.message_count + 1)
    counts = estimate.breakdown
    assert sum(asdict(counts).values()) == estimate.estimated_input_tokens
    assert all(value > 0 for value in asdict(counts).values())
    assert counts.history_tokens == counts.user_tokens + counts.assistant_tokens + counts.tool_result_tokens

    joined = estimator.estimate(request, payload=payload)
    assert joined.estimated_input_tokens == estimate.estimated_input_tokens
    assert joined.breakdown.project_memory_tokens == joined.breakdown.runtime_context_tokens == 0
    assert joined.breakdown.system_tokens >= sum((
        counts.system_tokens, counts.project_memory_tokens, counts.runtime_context_tokens,
    ))
    no_tools = request.model_copy(update={"tools": ()})
    omitted = estimator.estimate(no_tools, system_parts=parts)
    assert omitted.breakdown.tool_definition_tokens == 0
    assert omitted.estimated_input_tokens < estimate.estimated_input_tokens


def test_counts_actual_wire_only_not_original_provider_ids_or_invalid_arguments(projected_request):
    parts, request = projected_request
    estimator = TokenEstimator()
    old_call = request.messages[2].tool_calls[0]
    call = old_call.model_copy(update={"provider_call_id": "X" * 300})
    changed = request.model_copy(update={"messages": (
        *request.messages[:2], request.messages[2].model_copy(update={"tool_calls": (call,)}),
        request.messages[3],
    )})
    assert estimator.estimate(request, system_parts=parts) == estimator.estimate(changed, system_parts=parts)

    broken = call.model_copy(update={"arguments_raw": "not JSON " * 100})
    broken_request = changed.model_copy(update={"messages": (
        *changed.messages[:2], changed.messages[2].model_copy(update={"tool_calls": (broken,)}),
        changed.messages[3],
    )})
    safe_request = changed.model_copy(update={"messages": (
        *changed.messages[:2], changed.messages[2].model_copy(update={
            "tool_calls": (call.model_copy(update={"arguments_raw": "{}"}),),
        }), changed.messages[3],
    )})
    assert estimator.estimate(broken_request, system_parts=parts) == estimator.estimate(
        safe_request, system_parts=parts,
    )


def test_sdk_body_metadata_is_counted_without_counting_the_extra_body_wrapper(projected_request):
    parts, request = projected_request
    estimator = TokenEstimator()
    generic = estimator.estimate(request, system_parts=parts)
    payload = encode_chat_payload(request, model="synthetic-model", deepseek_thinking="disabled")
    estimate = estimator.estimate(request, system_parts=parts, payload=payload)
    direct = {key: value for key, value in payload.items() if key != "extra_body"}
    direct.update(payload["extra_body"])
    assert estimate == estimator.estimate(request, system_parts=parts, payload=direct)
    assert estimate.breakdown.other_metadata_tokens > generic.breakdown.other_metadata_tokens
    assert estimate.breakdown.history_tokens == generic.breakdown.history_tokens


@pytest.mark.parametrize("failure", ["type", "parts", "oversized", "unicode", "nonfinite", "missing-role"])
def test_estimation_failures_report_a_bounded_diagnostic_without_inventing_counts(projected_request, failure):
    parts, request = projected_request
    estimator = TokenEstimator()
    options = {}
    if failure == "type":
        request = []
    elif failure == "parts":
        options["system_parts"] = SystemPromptParts("Other prompt.")
    elif failure == "oversized":
        estimator = TokenEstimator(max_bytes=48)
    elif failure == "unicode":
        request = ChatRequest(messages=(ChatMessage(role="user", content="\ud800"),))
    elif failure == "nonfinite":
        options["payload"] = dict(encode_chat_payload(request), temperature=float("nan"))
    else:
        options["payload"] = encode_chat_payload(request)
        del options["payload"]["messages"][1]["role"]
    estimate = estimator.estimate(request, **options)
    assert estimate.status == "unavailable"
    assert estimate.estimated_input_tokens is None and estimate.breakdown is None
    assert estimate.diagnostic and len(estimate.diagnostic) < 200


def test_budget_and_provider_delta_have_distinct_meanings(projected_request):
    parts, request = projected_request
    estimate = TokenEstimator().estimate(request, system_parts=parts)
    policy = BudgetPolicy(model_context_window=1001, output_reserve=50)
    assert (policy.input_capacity, policy.high_watermark, policy.low_watermark) == (900, 765, 630)
    assert policy.output_reserve == 50 and policy.model_context_window == 1001
    snapshot = ContextMeasurementSnapshot(
        request_id=UUID(int=1), turn_id=UUID(int=2), attempt=1, trigger="initial",
        estimate=estimate, budget_policy=policy,
        actual_usage=TokenUsage(
            input_tokens=estimate.estimated_input_tokens + 7, output_tokens=4,
            total_tokens=estimate.estimated_input_tokens + 11,
        ),
    )
    assert snapshot.usage_status == "available" and snapshot.prompt_token_delta == 7
    assert snapshot.at_or_above_high_watermark is False
    absent = ContextMeasurementSnapshot(
        request_id=UUID(int=1), turn_id=UUID(int=2), attempt=1, trigger="initial", estimate=estimate,
    )
    assert absent.usage_status == "unavailable"
    assert absent.prompt_token_delta is None and absent.at_or_above_high_watermark is None
    with pytest.raises(ValueError, match="capacity"):
        BudgetPolicy(model_context_window=100, output_reserve=95)
