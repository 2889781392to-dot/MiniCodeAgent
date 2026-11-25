"""Choose a measured request projection in deterministic layer order."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Literal, Protocol

from ..chat_payload import encode_chat_payload
from ..protocol import ChatMessage, ChatRequest, ModelError, ModelReply, ToolSpec
from .budget import BudgetPolicy, SystemPromptParts, TokenEstimate, TokenEstimator
from .builder import (
    ArchiveBounds, ContextBuilder, ContextProjection, l1_replacements, l2_replacements,
    l3_bounds, l3_checkpoint, l4_source, projected_event_seqs, render_checkpoint,
)
from .compression.l4 import SummaryBudget, cited_items, parse_handoff
from .events import (
    Checkpoint, CompactionCommittedPayload, CompactionLevel, CompactionLevels, ConversationHistory,
    HandoffCheckpoint, TextReplacement,
)


CompactionStatus = Literal[
    "not_needed", "target_reached", "reduced_above_target", "protected",
    "invalid_candidate", "extraction_failed", "insufficient_evidence", "no_gain",
    "estimate_unavailable", "candidate_over_capacity", "summary_input_limit", "summary_output_limit",
    "summary_failed", "summary_cached",
]


class Summarizer(Protocol):
    def __call__(self, request: ChatRequest, *, max_output_tokens: int) -> ModelReply: ...


class SummaryEncoder(Protocol):
    def __call__(self, request: ChatRequest, *, max_output_tokens: int) -> dict: ...


@dataclass(frozen=True, slots=True)
class SummaryDiagnostic:
    input_limit: int
    output_limit: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    handoff_tokens: int | None = None
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    called: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LayerDiagnostic:
    level: CompactionLevel
    status: CompactionStatus
    triggered: bool
    original_tokens: int | None
    candidate_tokens: int | None = None
    saved_tokens: int | None = None
    target_seqs: tuple[int, ...] = ()
    selected: bool = False


@dataclass(frozen=True, slots=True)
class CompactionDiagnostic:
    status: CompactionStatus
    original_tokens: int | None
    candidate_tokens: int | None
    selected_tokens: int | None
    can_send: bool
    layers: tuple[LayerDiagnostic, ...] = ()
    archive: ArchiveBounds | None = None
    summary: SummaryDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class PreparedContext:
    projection: ContextProjection
    request: ChatRequest
    compaction: CompactionCommittedPayload | None
    diagnostic: CompactionDiagnostic


@dataclass(frozen=True, slots=True)
class _View:
    projection: ContextProjection
    request: ChatRequest
    tokens: int
    replacements: tuple[TextReplacement, ...] = ()
    levels: CompactionLevels = ()


@dataclass(frozen=True, slots=True)
class _Window:
    capacity: int
    target: int

    def shrink_status(self, tokens: int) -> CompactionStatus:
        if tokens <= self.target:
            return "target_reached"
        if tokens <= self.capacity:
            return "reduced_above_target"
        return "candidate_over_capacity"


class ContextWindowManager:
    def __init__(
        self, builder: ContextBuilder, estimator: TokenEstimator, policy: BudgetPolicy,
        *, encode: Callable[[ChatRequest], dict] = encode_chat_payload, enable_checkpoint: bool = False,
        summarizer: Summarizer | None = None, summary_budget: SummaryBudget = SummaryBudget(),
        encode_summary: SummaryEncoder | None = None,
    ) -> None:
        self._builder = builder
        self._estimator = estimator
        self._policy = policy
        self._encode = encode
        self._enable_checkpoint = enable_checkpoint or summarizer is not None
        self._summarizer = summarizer
        self._summary_budget = summary_budget
        self._encode_summary = encode_summary if encode_summary is not None else self._summary_payload
        self._unsuccessful_summaries: deque[str] = deque(maxlen=64)

    def _summary_payload(self, request: ChatRequest, *, max_output_tokens: int) -> dict:
        return self._encode(request) | {"max_tokens": max_output_tokens, "tool_choice": "none"}

    def prepare(
        self, history: ConversationHistory, projection: ContextProjection,
        system_parts: SystemPromptParts, tools: tuple[ToolSpec, ...], *, rejected_tokens: int | None = None,
    ) -> PreparedContext:
        request = ChatRequest(messages=projection.messages, tools=tools)
        before = self._tokens(request, system_parts)
        capacity = self._policy.input_capacity
        target = self._policy.low_watermark
        if rejected_tokens is not None:
            # The rejected estimate is an exclusive bound. Keep the existing
            # low-watermark ratio on the smaller effective capacity.
            capacity = min(capacity, rejected_tokens - 1)
            target = capacity * 70 // 100
        window = _Window(capacity, target)
        layers: list[LayerDiagnostic] = []
        archive = None
        summary = None
        strategies: tuple[tuple[CompactionLevel, Callable[[ConversationHistory], tuple[TextReplacement, ...]]], ...] = (
            ("L1", l1_replacements), ("L2", l2_replacements),
        )

        def original(status: CompactionStatus, candidate_tokens: int | None = None) -> PreparedContext:
            can_send = rejected_tokens is None and (before is None or before <= capacity)
            diagnostic = CompactionDiagnostic(
                status, before, candidate_tokens, before if can_send else None, can_send, tuple(layers), archive, summary,
            )
            return PreparedContext(projection, request, None, diagnostic)

        if before is None or (rejected_tokens is None and before < self._policy.high_watermark):
            status = "estimate_unavailable" if before is None else "not_needed"
            layers.extend(LayerDiagnostic(level, status, False, before) for level, _ in strategies)
            if self._enable_checkpoint:
                layers.append(LayerDiagnostic("L3", status, False, before))
            if self._summarizer is not None:
                layers.append(LayerDiagnostic("L4", status, False, before))
            return original(status)

        current = _View(projection, request, before)
        for level, propose in strategies:
            if level == "L2" and current.tokens <= window.target:
                layers.append(LayerDiagnostic(level, "not_needed", False, current.tokens))
                continue
            current, observation = self._try_layer(history, projection, system_parts, current, level, propose, window)
            layers.append(observation)

        if self._enable_checkpoint:
            if current.tokens <= window.target:
                layers.append(LayerDiagnostic("L3", "not_needed", False, current.tokens))
            else:
                current, observation, archive = self._try_checkpoint(history, system_parts, current, window)
                layers.append(observation)

        if self._summarizer is not None:
            if current.tokens <= window.target:
                layers.append(LayerDiagnostic("L4", "not_needed", False, current.tokens))
            elif archive is None or archive.covered_through_seq is None:
                layers.append(LayerDiagnostic("L4", "protected", False, current.tokens))
            else:
                current, observation, summary = self._try_handoff(history, system_parts, current, archive, window)
                layers.append(observation)

        if not current.levels:
            failed = next((layer for layer in reversed(layers) if layer.status != "protected"), layers[0])
            return original(failed.status, failed.candidate_tokens)
        if current.tokens > capacity:
            return original("candidate_over_capacity", current.tokens)

        compaction = CompactionCommittedPayload(
            base_seq=history[-1].seq, levels=current.levels, before_tokens=before,
            after_tokens=current.tokens, replacements=current.replacements,
            checkpoint=current.projection.checkpoint if {"L3", "L4"}.intersection(current.levels) else None,
            trigger="threshold" if rejected_tokens is None else "overflow",
        )
        selected = tuple(replace(layer, selected=layer.level in current.levels) for layer in layers)
        return PreparedContext(current.projection, current.request, compaction, CompactionDiagnostic(
            window.shrink_status(current.tokens), before, current.tokens, current.tokens, True, selected, archive, summary,
        ))

    def _try_layer(
        self, history: ConversationHistory, original: ContextProjection, system_parts: SystemPromptParts,
        current: _View, level: CompactionLevel, propose: Callable[[ConversationHistory], tuple[TextReplacement, ...]],
        window: _Window,
    ) -> tuple[_View, LayerDiagnostic]:
        diagnostic = LayerDiagnostic(level, "protected", True, current.tokens)
        try:
            added = propose(history)
        except Exception:
            return current, replace(diagnostic, status="extraction_failed")
        if not added:
            return current, diagnostic

        levels = (*current.levels, level)
        try:
            replacements = (*current.replacements, *added)
            diagnostic = replace(diagnostic, target_seqs=tuple(item.target_seq for item in added))
            # L1 keeps the existing preview call shape.
            options = {} if levels == ("L1",) else {"levels": levels}
            projection = self._builder.project(history, system_parts.text, replacements=replacements, **options)
            if not _matches_replacements(history, original, projection, replacements):
                return current, replace(diagnostic, status="invalid_candidate")
            request = ChatRequest(messages=projection.messages, tools=current.request.tools)
        except Exception:
            return current, replace(diagnostic, status="invalid_candidate")

        return self._evaluate(current, projection, request, replacements, levels, diagnostic, system_parts, window)

    def _try_checkpoint(
        self, history: ConversationHistory, system_parts: SystemPromptParts, current: _View, window: _Window,
    ) -> tuple[_View, LayerDiagnostic, ArchiveBounds | None]:
        diagnostic = LayerDiagnostic("L3", "protected", True, current.tokens)
        archive = None
        try:
            archive = l3_bounds(history)
            if archive.covered_through_seq is None:
                return current, diagnostic, archive
            diagnostic = replace(diagnostic, target_seqs=(archive.covered_through_seq,))
            checkpoint = l3_checkpoint(history)
        except Exception:
            return current, replace(diagnostic, status="extraction_failed"), archive
        if checkpoint is None:
            return current, replace(diagnostic, status="insufficient_evidence"), archive

        try:
            if checkpoint.covered_through_seq != archive.covered_through_seq:
                return current, replace(diagnostic, status="invalid_candidate"), archive
            projection, request, replacements, levels = self._checkpoint_preview(history, system_parts, current, checkpoint)
        except Exception:
            return current, replace(diagnostic, status="invalid_candidate"), archive

        selected, diagnostic = self._evaluate(
            current, projection, request, replacements, levels, diagnostic, system_parts, window,
        )
        return selected, diagnostic, archive

    def _checkpoint_preview(
        self, history: ConversationHistory, system_parts: SystemPromptParts, current: _View, checkpoint: Checkpoint,
    ) -> tuple[ContextProjection, ChatRequest, tuple[TextReplacement, ...], CompactionLevels]:
        replacements = tuple(item for item in current.replacements if item.target_seq > checkpoint.covered_through_seq)
        surviving = {"L2" if history[item.target_seq - 1].type == "tool_result" else "L1" for item in replacements}
        levels = (*tuple(level for level in current.levels if level in surviving),
                  "L4" if checkpoint.kind == "handoff" else "L3")
        projection = self._builder.project(
            history, system_parts.text, replacements=replacements, levels=levels, checkpoint=checkpoint,
        )
        if not _matches_checkpoint(history, current.projection, projection, checkpoint):
            raise ValueError("Checkpoint preview changed protected messages or calls.")
        return projection, ChatRequest(messages=projection.messages, tools=current.request.tools), replacements, levels

    def _try_handoff(
        self, history: ConversationHistory, system_parts: SystemPromptParts, current: _View, archive: ArchiveBounds,
        window: _Window,
    ) -> tuple[_View, LayerDiagnostic, SummaryDiagnostic]:
        end = archive.covered_through_seq
        assert self._summarizer is not None and end is not None
        budget = self._summary_budget
        summary = SummaryDiagnostic(budget.input_tokens, budget.output_tokens)
        layer = LayerDiagnostic("L4", "insufficient_evidence", True, current.tokens, target_seqs=(end,))
        key = None

        def reject(status: CompactionStatus, reason: str) -> tuple[_View, LayerDiagnostic, SummaryDiagnostic]:
            if key is not None and key not in self._unsuccessful_summaries:
                self._unsuccessful_summaries.append(key)
            return current, replace(layer, status=status), replace(summary, failure_reason=reason)

        try:
            tail = _checkpoint_tail(history, current.projection, end)
            floor = ChatRequest(messages=(current.request.messages[0], *tail), tools=current.request.tools)
        except (IndexError, ValueError):
            return reject("invalid_candidate", "Could not identify the protected main-request content.")
        floor_tokens = self._tokens(floor, system_parts)
        if floor_tokens is None:
            return reject("estimate_unavailable", "Protected main-request content could not be measured.")
        if floor_tokens > window.capacity:
            return reject("candidate_over_capacity", "Protected main-request content alone exceeds capacity.")

        try:
            source = l4_source(history, end)
        except Exception:
            return reject("extraction_failed", "Could not prepare validated summary evidence.")
        if source is None:
            return reject("insufficient_evidence", "No source evidence or current goal anchor.")
        request, references = source
        try:
            payload = self._encode_summary(request, max_output_tokens=budget.output_tokens)
            goal_event = next(event for event in reversed(history) if event.type == "user_message")
            fingerprint = (history[end - 1].event_id.hex, goal_event.event_id.hex, 1, budget.input_tokens,
                           window.capacity, window.target, payload)
            key = sha256(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            if key in self._unsuccessful_summaries:
                return reject("summary_cached", "An identical source and request policy already failed.")
            estimate = self._estimator.estimate(request, payload=payload)
            count = estimate.estimated_input_tokens
        except Exception:
            return reject("estimate_unavailable", "Summary input estimate is unavailable.")
        if type(count) is not int or count <= 0:
            return reject("estimate_unavailable", "Summary input estimate is unavailable.")
        summary = replace(summary, input_tokens=count)
        if count > budget.input_tokens:
            return reject("summary_input_limit", "Summary input exceeds its independent limit.")

        summary = replace(summary, called=True)
        try:
            reply = self._summarizer(request, max_output_tokens=budget.output_tokens)
        except ModelError as exc:
            return reject("summary_failed", f"Summary model failed ({exc.kind}).")
        except Exception:
            return reject("summary_failed", "Summary model failed.")
        if not isinstance(reply, ModelReply) or reply.finish_reason != "stop" or reply.tool_calls or not reply.content.strip():
            return reject("invalid_candidate", "Expected a complete, tools-free summary reply.")
        if reply.usage is not None:
            summary = replace(summary, provider_input_tokens=reply.usage.input_tokens,
                              provider_output_tokens=reply.usage.output_tokens)
            if reply.usage.output_tokens > budget.output_tokens:
                return reject("summary_output_limit", "Provider output usage exceeds the summary limit.")
        try:
            output_tokens = self._estimator.estimate_text(reply.content)
        except Exception:
            return reject("estimate_unavailable", "Summary output estimate is unavailable.")
        summary = replace(summary, output_tokens=output_tokens)
        if output_tokens > budget.output_tokens:
            return reject("summary_output_limit", "Summary output estimate exceeds its limit.")
        try:
            handoff = parse_handoff(reply.content, references=references)
            if not set(cited_items(handoff)).issubset(references.values()):
                return reject("invalid_candidate", "Handoff cites evidence outside the provided source.")
            checkpoint = HandoffCheckpoint(covered_through_seq=end, kind="handoff", summary=handoff)
            handoff_tokens = self._estimator.estimate_text(handoff.model_dump_json(exclude_defaults=True))
            summary = replace(summary, handoff_tokens=handoff_tokens)
            projection, projected_request, replacements, levels = self._checkpoint_preview(
                history, system_parts, current, checkpoint,
            )
        except Exception:
            return reject("invalid_candidate", "Handoff failed structure, source, or protected projection validation.")

        view, layer = self._evaluate(current, projection, projected_request, replacements, levels, layer, system_parts,
                                     window)
        if view is current or view.tokens > window.capacity:
            return reject(layer.status, "Handoff did not produce a smaller sendable main request.")
        return view, layer, summary

    def _evaluate(
        self, current: _View, projection: ContextProjection, request: ChatRequest,
        replacements: tuple[TextReplacement, ...], levels: CompactionLevels, diagnostic: LayerDiagnostic,
        system_parts: SystemPromptParts, window: _Window,
    ) -> tuple[_View, LayerDiagnostic]:
        after = self._tokens(request, system_parts)
        if after is None:
            return current, replace(diagnostic, status="estimate_unavailable")
        diagnostic = replace(diagnostic, candidate_tokens=after, saved_tokens=current.tokens - after)
        if after >= current.tokens:
            return current, replace(diagnostic, status="no_gain")
        return _View(projection, request, after, replacements, levels), replace(
            diagnostic, status=window.shrink_status(after),
        )

    def _tokens(self, request: ChatRequest, parts: SystemPromptParts) -> int | None:
        try:
            estimate = self._estimator.estimate(request, system_parts=parts, payload=self._encode(request))
        except Exception:
            # No measured input count is preferable to inventing one on observation failure.
            return None
        count = estimate.estimated_input_tokens if isinstance(estimate, TokenEstimate) else None
        return count if type(count) is int and count > 0 else None


def _matches_replacements(
    history: ConversationHistory, original: ContextProjection, candidate: ContextProjection,
    replacements: tuple[TextReplacement, ...],
) -> bool:
    # The Builder validates eligibility at each source seq. Check the resulting
    # projection contains exactly those text edits, preserving every other field.
    seqs = projected_event_seqs(history)
    header = 1 if original.checkpoint is None else 2
    if (
        len(original.messages) != len(seqs) + header
        or len(candidate.messages) != len(original.messages)
        or original.messages[:header] != candidate.messages[:header]
        or original.consumed_call_ids != candidate.consumed_call_ids
        or original.checkpoint != candidate.checkpoint
    ):
        return False
    edits = {item.target_seq: item.content for item in replacements}
    for seq, old, new in zip(seqs, original.messages[header:], candidate.messages[header:], strict=True):
        if seq in edits:
            old = old.model_copy(update={"content": edits.pop(seq)})
        if old != new:
            return False
    return not edits


def _checkpoint_tail(history: ConversationHistory, current: ContextProjection, end: int) -> tuple[ChatMessage, ...]:
    header = 1 if current.checkpoint is None else 2
    covered = current.checkpoint.covered_through_seq if current.checkpoint is not None else 0
    return tuple(
        message for seq, message in zip(projected_event_seqs(history, covered), current.messages[header:], strict=True)
        if seq > end
    )


def _matches_checkpoint(
    history: ConversationHistory, current: ContextProjection, candidate: ContextProjection,
    checkpoint: Checkpoint,
) -> bool:
    tail = _checkpoint_tail(history, current, checkpoint.covered_through_seq)
    return (
        candidate.checkpoint == checkpoint
        and candidate.messages == (
            current.messages[0], ChatMessage(role="user", content=render_checkpoint(checkpoint)), *tail,
        )
        and candidate.consumed_call_ids == tuple(message.call_id for message in tail if message.role == "tool")
    )
