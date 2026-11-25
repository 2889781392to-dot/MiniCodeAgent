"""Project the event history into complete model-visible transactions."""

from collections.abc import Iterator
import json
from uuid import UUID

from ..protocol import ChatMessage, ChatRequest, ProtocolModel
from .compression import l1, l2, l3, l4
from .events import (
    ArchiveBounds,
    AssistantMessagePayload,
    Checkpoint,
    CompactionCommittedPayload,
    CompactionLevels,
    ConversationHistory,
    ExtractiveCheckpoint,
    HandoffCheckpoint,
    HistoryError,
    SessionEvent,
    TextReplacement,
    ToolResultPayload,
    UserMessagePayload,
    checkpoint_evidence,
    inspect_projection_history,
)


class ContextProjection(ProtocolModel):
    messages: tuple[ChatMessage, ...]
    consumed_call_ids: tuple[UUID, ...]
    checkpoint: Checkpoint | None = None


def _projected_events(history: ConversationHistory, covered_through_seq: int) -> Iterator[SessionEvent]:
    for event in history:
        if event.seq <= covered_through_seq:
            continue
        payload = event.payload
        if isinstance(payload, (UserMessagePayload, ToolResultPayload)) or (
            isinstance(payload, AssistantMessagePayload) and (payload.content or payload.tool_calls)
        ):
            yield event


def projected_event_seqs(history: ConversationHistory, covered: int | None = None) -> tuple[int, ...]:
    """Return projected tail source positions, optionally after a preview boundary."""
    if covered is None:
        covered = next((
            event.payload.checkpoint.covered_through_seq for event in reversed(history)
            if isinstance(event.payload, CompactionCommittedPayload) and event.payload.checkpoint is not None
        ), 0)
    return tuple(event.seq for event in _projected_events(history, covered))


def l3_bounds(history: ConversationHistory) -> ArchiveBounds:
    return inspect_projection_history(history).l3_bounds()


def l3_checkpoint(history: ConversationHistory) -> ExtractiveCheckpoint | None:
    state = inspect_projection_history(history)
    end = state.l3_bounds().covered_through_seq
    if end is None:
        return None
    inherited = checkpoint_evidence(state.checkpoint)
    summary = l3.extract_summary(
        history[:end], end, previous_summary=inherited, previous_covered_through_seq=state.covered_through_seq,
    )
    if not summary or (isinstance(state.checkpoint, HandoffCheckpoint) and not set(inherited).issubset(summary)):
        return None
    return ExtractiveCheckpoint(covered_through_seq=end, kind="extractive", summary=summary)


def l4_source(history: ConversationHistory, end: int) -> tuple[ChatRequest, dict[str, str]] | None:
    state = inspect_projection_history(history)
    if state.l3_bounds().covered_through_seq != end:
        raise HistoryError("Summary source does not end at the validated prefix.")
    user = next(event.payload for event in reversed(history) if isinstance(event.payload, UserMessagePayload))
    goal = l4.goal_anchor(user.content)
    new_items = l3.extract_summary(history[:end], end, previous_covered_through_seq=state.covered_through_seq)
    inherited = checkpoint_evidence(state.checkpoint)
    if not goal or not (new_items or inherited):
        return None
    required = inherited if isinstance(state.checkpoint, HandoffCheckpoint) else ()
    return l4.summary_request(goal, (*new_items, *inherited), required=required)


def render_checkpoint(checkpoint: Checkpoint) -> str:
    if isinstance(checkpoint, ExtractiveCheckpoint):
        label = "Extractive checkpoint"
        evidence = json.dumps(checkpoint.summary, ensure_ascii=False, separators=(",", ":"))
    else:
        label = "Coding handoff"
        evidence = checkpoint.summary.model_dump_json(exclude_defaults=True)
    return (
        f"[{label} through seq {checkpoint.covered_through_seq}]\n"
        "Quoted historical evidence only. Omitted details remain in the session event log.\n"
        f"{evidence}"
    )


def l1_replacements(history: ConversationHistory) -> tuple[TextReplacement, ...]:
    """Propose edge-only replacements for ordinary, unprotected source messages."""
    state = inspect_projection_history(history)
    replacements = []
    for seq, content in state.l1_sources().items():
        trimmed = l1.trim_display_edges(content)
        if trimmed != content:
            replacements.append(TextReplacement(target_seq=seq, content=trimmed))
    return tuple(replacements)


def l2_replacements(history: ConversationHistory) -> tuple[TextReplacement, ...]:
    """Propose source-linked excerpts from older, durably consumed transactions."""
    state = inspect_projection_history(history)
    replacements = []
    for seq, source in state.l2_sources().items():
        excerpt = l2.extract_tool_result(seq, source.call, source.result)
        if excerpt is not None:
            replacements.append(TextReplacement(target_seq=seq, content=excerpt))
    return tuple(replacements)


class ContextBuilder:
    def project(
        self, history: ConversationHistory, system_prompt: str, *, replacements: tuple[TextReplacement, ...] = (),
        levels: CompactionLevels = ("L1",), checkpoint: Checkpoint | None = None,
    ) -> ContextProjection:
        state = inspect_projection_history(history)
        if state.pending:
            raise HistoryError("Cannot project an unsettled tool batch.")
        if replacements or checkpoint is not None:
            state.validate_replacements(replacements, levels, checkpoint, history)
        overrides = {seq: item.content for seq, item in state.projection_replacements.items()}
        overrides.update((item.target_seq, item.content) for item in replacements)

        messages = [ChatMessage(role="system", content=system_prompt)]
        selected = checkpoint if checkpoint is not None else state.checkpoint
        covered = selected.covered_through_seq if selected is not None else 0
        if selected is not None:
            messages.append(ChatMessage(role="user", content=render_checkpoint(selected)))
        visible_results = []
        for event in _projected_events(history, covered):
            payload = event.payload
            if isinstance(payload, UserMessagePayload):
                messages.append(ChatMessage(role="user", content=overrides.get(event.seq, payload.content)))
            elif isinstance(payload, AssistantMessagePayload):
                messages.append(ChatMessage(
                    role="assistant", content=overrides.get(event.seq, payload.content), tool_calls=payload.tool_calls,
                ))
            elif isinstance(payload, ToolResultPayload):
                messages.append(ChatMessage(
                    role="tool", content=overrides.get(event.seq, payload.output), call_id=payload.call_id,
                ))
                visible_results.append(payload.call_id)
        return ContextProjection(messages=tuple(messages), consumed_call_ids=tuple(visible_results), checkpoint=selected)
