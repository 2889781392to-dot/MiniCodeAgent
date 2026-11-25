"""Budget selection for L1, independent of the runtime and event writer."""

from copy import deepcopy
from uuid import UUID

import pytest

from minicodeagent.chat_payload import encode_chat_payload
from minicodeagent.context.budget import BudgetPolicy, SystemPromptParts, TokenEstimate, TokenEstimator
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.compression.l1 import is_display_edge_trim, trim_display_edges
from minicodeagent.context.events import AssistantMessagePayload, TurnFinishedPayload, UserMessagePayload
from minicodeagent.context.manager import ContextWindowManager
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.protocol import ChatRequest, ToolSpec


@pytest.mark.parametrize(("source", "trimmed"), [
    ("\n\r\n  first\n\n  second\r\n\n\r\n", "  first\n\n  second\r\n"),
    ("\n\npython -m pytest\n\n", "python -m pytest\n"),
    ("\n\n\r\n", "\n\n\r\n"),
    ("\r\n \t\r\n", "\r\n \t\r\n"),
    ("\n\n```py\n    value = 1\n\n", "\n\n```py\n    value = 1\n\n"),
    ("\n~~~bash\n    python -m pytest\n~~~\n\n", "\n~~~bash\n    python -m pytest\n~~~\n\n"),
    ("\n  indent then spaces  \r\n", "  indent then spaces  \r\n"),
    ("plain\n", "plain\n"),
])
def test_l1_removes_only_physically_empty_edge_lines(source, trimmed):
    assert trim_display_edges(source) == trimmed
    assert trim_display_edges(trimmed) == trimmed
    assert is_display_edge_trim(source, trimmed) is (source != trimmed)
    assert not is_display_edge_trim(source, trimmed + " changed")


@pytest.fixture
def logical_request():
    writer = SessionEventWriter(session_id=UUID(int=1))
    old, current = UUID(int=2), UUID(int=3)
    writer.append("user_message", UserMessagePayload(
        turn_id=old, content="\n" * 240 + "  Verify the code and preserve its public API.\r\n" + "\r\n" * 150,
    ))
    writer.append("assistant_message", AssistantMessagePayload(
        turn_id=old, request_id=UUID(int=4), content="Earlier answer, with its source evidence.", finish_reason="stop",
    ))
    writer.append("turn_finished", TurnFinishedPayload(
        turn_id=old, status="completed", reason="model_stop", assistant_seq=2,
    ))
    writer.append("user_message", UserMessagePayload(turn_id=current, content="\n\nKeep the latest goal.\n\n"))
    parts = SystemPromptParts("Stable system.", "\n\nProject rules.", "\nPython 3.11.")
    tools = (ToolSpec(name="read_file", description="Inspect file contents.", parameters={"type": "object"}),)
    builder = ContextBuilder()
    projection = builder.project(writer.history, parts.text)
    return writer, builder, projection, parts, tools


def policy_with_capacity(capacity):
    policy = BudgetPolicy(model_context_window=(100 * capacity + 94) // 95, output_reserve=0)
    assert policy.input_capacity == capacity
    return policy


def test_selected_l1_view_is_reencoded_at_exact_high_watermark_without_mutation(logical_request):
    writer, builder, original, parts, tools = logical_request
    history = writer.history
    requests = []

    def encode(request):
        payload = encode_chat_payload(request, model="synthetic", deepseek_thinking="disabled")
        requests.append(deepcopy(payload))
        return payload

    estimate = TokenEstimator()
    initial = ChatRequest(messages=original.messages, tools=tools)
    old_payload = encode(initial)
    old_tokens = estimate.estimate(initial, system_parts=parts, payload=old_payload).estimated_input_tokens
    policy = policy_with_capacity((old_tokens * 100 + 84) // 85)
    assert old_tokens == policy.high_watermark
    original_json = tuple(event.model_dump(mode="json") for event in history)

    prepared = ContextWindowManager(builder, estimate, policy, encode=encode).prepare(history, original, parts, tools)
    candidate_payload = requests[-1]
    assert requests[:-1] == [old_payload, old_payload]
    assert candidate_payload == encode_chat_payload(prepared.request, model="synthetic", deepseek_thinking="disabled")
    assert {k: v for k, v in old_payload.items() if k != "messages"} == {
        k: v for k, v in candidate_payload.items() if k != "messages"
    }
    assert candidate_payload["messages"][0] == old_payload["messages"][0]
    assert candidate_payload["messages"][2:] == old_payload["messages"][2:]
    assert prepared.request.messages[1].content == "  Verify the code and preserve its public API.\r\n"
    assert prepared.projection.consumed_call_ids == original.consumed_call_ids
    selected = estimate.estimate(prepared.request, system_parts=parts, payload=candidate_payload).estimated_input_tokens
    assert selected < old_tokens and selected <= policy.low_watermark
    assert prepared.diagnostic.can_send and prepared.diagnostic.status == "target_reached"
    assert prepared.diagnostic.original_tokens == old_tokens
    assert prepared.diagnostic.candidate_tokens == prepared.diagnostic.selected_tokens == selected
    assert prepared.compaction.base_seq == 4
    assert [(item.target_seq, item.content) for item in prepared.compaction.replacements] == [
        (1, prepared.request.messages[1].content),
    ]
    assert (prepared.compaction.before_tokens, prepared.compaction.after_tokens) == (old_tokens, selected)
    assert tuple(event.model_dump(mode="json") for event in writer.history) == original_json
    assert original.messages[1].content == history[0].payload.content


class ControlledEstimator:
    """Drive exact watermarks without depending on sample text lengths."""

    def __init__(self, *counts):
        self.counts = iter(counts)

    def estimate(self, request, **_kwargs):
        count = next(self.counts)
        if isinstance(count, Exception):
            raise count
        return TokenEstimate(message_count=len(request.messages), estimated_input_tokens=count)


@pytest.mark.parametrize(("counts", "status", "can_send", "selected"), [
    ((84,), "not_needed", True, 84),
    ((100, 70), "target_reached", True, 70),
    ((101, 96), "reduced_above_target", True, 96),
    ((101, 100), "reduced_above_target", True, 100),
    ((95, 95), "no_gain", True, 95),
    ((120, 121), "no_gain", False, None),
    ((120, 110), "candidate_over_capacity", False, None),
    ((None,), "estimate_unavailable", True, None),
    ((RuntimeError("estimation failed"),), "estimate_unavailable", True, None),
    ((95, None), "estimate_unavailable", True, 95),
    ((120, None), "estimate_unavailable", False, None),
])
def test_manager_distinguishes_fallback_savings_and_hard_capacity(logical_request, counts, status, can_send, selected):
    writer, builder, original, parts, tools = logical_request
    manager = ContextWindowManager(builder, ControlledEstimator(*counts), policy_with_capacity(100))
    prepared = manager.prepare(writer.history, original, parts, tools)
    diagnostic = prepared.diagnostic
    assert (diagnostic.status, diagnostic.can_send, diagnostic.selected_tokens) == (status, can_send, selected)
    assert (prepared.compaction is not None) is (status in ("target_reached", "reduced_above_target"))
    if prepared.compaction is None:
        assert prepared.projection is original
        assert prepared.request.messages == original.messages


def test_protected_history_below_hard_limit_falls_back_and_above_it_cannot_send(logical_request):
    writer, builder, _, parts, tools = logical_request
    # This turn has only its latest user, which is protected despite the removable-looking edges.
    current = SessionEventWriter(session_id=UUID(int=5))
    current.append("user_message", writer.history[-1].payload)
    history = current.history
    original = builder.project(history, parts.text)
    for before, send in ((94, True), (121, False)):
        manager = ContextWindowManager(builder, ControlledEstimator(before), policy_with_capacity(100))
        prepared = manager.prepare(history, original, parts, tools)
        assert prepared.diagnostic.status == "protected"
        assert prepared.diagnostic.can_send is send
        assert prepared.diagnostic.candidate_tokens is None and prepared.compaction is None
        assert prepared.projection is original


@pytest.mark.parametrize("tamper", ["system", "protected", "structure", "result_ids"])
def test_candidate_tampering_cannot_change_protected_content_or_request_structure(logical_request, tamper):
    writer, _, original, parts, tools = logical_request

    class BadBuilder(ContextBuilder):
        def project(self, history, system_prompt, *, replacements=()):
            view = super().project(history, system_prompt, replacements=replacements)
            if tamper == "structure":
                return view.model_copy(update={"messages": view.messages[:-1]})
            if tamper == "result_ids":
                return view.model_copy(update={"consumed_call_ids": (UUID(int=17),)})
            index = 0 if tamper == "system" else -1
            messages = list(view.messages)
            messages[index] = messages[index].model_copy(update={"content": "modified"})
            return view.model_copy(update={"messages": tuple(messages)})

    manager = ContextWindowManager(BadBuilder(), ControlledEstimator(95), policy_with_capacity(100))
    prepared = manager.prepare(writer.history, original, parts, tools)
    assert prepared.diagnostic.status == "invalid_candidate" and prepared.diagnostic.can_send
    assert prepared.diagnostic.candidate_tokens is None
    assert prepared.projection is original and prepared.compaction is None
