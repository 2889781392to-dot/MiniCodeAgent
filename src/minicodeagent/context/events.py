"""Events and the rules for accepting and closing tool transactions."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import ntpath
from pathlib import PurePosixPath, PureWindowsPath
import posixpath
from typing import Annotated, Literal, NamedTuple
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from ..protocol import ProtocolModel, TokenUsage, ToolCall, ToolResult, TurnStatus
from .compression import l1, l2, l3, l4

EventType = Literal[
    "session_created", "user_message", "model_request_started", "assistant_message",
    "tool_result", "turn_finished", "compaction_committed",
]
ModelRequestTrigger = Literal["initial", "after_tools", "resume", "retry_transient", "retry_overflow"]
CompactionLevel = Literal["L1", "L2", "L3", "L4"]
CompactionLevels = tuple[CompactionLevel, ...]
_COMPACTION_LAYER_ORDER = ("L1", "L2", "L3", "L4")


def _valid_compaction_levels(levels: CompactionLevels) -> bool:
    return bool(levels) and levels == tuple(level for level in _COMPACTION_LAYER_ORDER if level in levels)


class SessionCreatedPayload(ProtocolModel):
    title: str
    workspace: str
    parent_session_id: UUID | None = None
    parent_through_seq: int | None = Field(default=None, ge=1)

    @field_validator("workspace")
    @classmethod
    def check_workspace_path(cls, value: str) -> str:
        # Check the stored spelling without consulting the live filesystem.
        windows_path = PureWindowsPath(value).is_absolute() and ntpath.normpath(value) == value
        posix_path = PurePosixPath(value).is_absolute() and posixpath.normpath(value) == value
        if "\0" in value or not (windows_path or posix_path):
            raise ValueError("Workspace must be a normalized absolute path.")
        return value

    @model_validator(mode="after")
    def check_parent_reference(self) -> "SessionCreatedPayload":
        if (self.parent_session_id is None) != (self.parent_through_seq is None):
            raise ValueError("Parent session ID and sequence must be provided together.")
        return self


class UserMessagePayload(ProtocolModel):
    turn_id: UUID
    content: str


class ModelRequestStartedPayload(ProtocolModel):
    turn_id: UUID
    request_id: UUID
    attempt: int = Field(ge=1)
    trigger: ModelRequestTrigger


class AssistantMessagePayload(ProtocolModel):
    turn_id: UUID
    request_id: UUID
    content: str
    finish_reason: str
    tool_calls: tuple[ToolCall, ...] = ()
    consumed_call_ids: tuple[UUID, ...] = ()
    usage: TokenUsage | None = None
    rejected_tool_calls: tuple[ToolCall, ...] = ()


class ToolResultPayload(ToolResult):
    turn_id: UUID


class TurnFinishedPayload(ProtocolModel):
    turn_id: UUID
    status: TurnStatus
    reason: str
    assistant_seq: int | None = Field(default=None, ge=1)


class TextReplacement(ProtocolModel):
    target_seq: int = Field(ge=1)
    content: str


class ExtractiveCheckpoint(ProtocolModel):
    covered_through_seq: int = Field(ge=1)
    kind: Literal["extractive"]
    summary: tuple[str, ...] = Field(min_length=1)


class HandoffCheckpoint(ProtocolModel):
    covered_through_seq: int = Field(ge=1)
    kind: Literal["handoff"]
    summary: l4.CodingHandoff


Checkpoint = Annotated[ExtractiveCheckpoint | HandoffCheckpoint, Field(discriminator="kind")]


def checkpoint_evidence(checkpoint: Checkpoint | None) -> tuple[str, ...]:
    if checkpoint is None:
        return ()
    if isinstance(checkpoint, ExtractiveCheckpoint):
        return checkpoint.summary
    return l4.cited_items(checkpoint.summary)


def _checkpoint_level(checkpoint: Checkpoint | None) -> CompactionLevels:
    if checkpoint is None:
        return ()
    return ("L3",) if checkpoint.kind == "extractive" else ("L4",)


def _matches_checkpoint_level(levels: CompactionLevels, checkpoint: Checkpoint | None) -> bool:
    return tuple(level for level in levels if level in ("L3", "L4")) == _checkpoint_level(checkpoint)


class CompactionCommittedPayload(ProtocolModel):
    """The selected text overlays and optional history checkpoint."""

    base_seq: int = Field(ge=1)
    policy_version: int = Field(default=1, ge=1, le=1)
    levels: CompactionLevels = ("L1",)
    trigger: Literal["threshold", "overflow"] = "threshold"
    before_tokens: int = Field(ge=1)
    after_tokens: int = Field(ge=1)
    replacements: tuple[TextReplacement, ...]
    checkpoint: Checkpoint | None = None

    @model_validator(mode="after")
    def check_compaction(self) -> "CompactionCommittedPayload":
        if not _valid_compaction_levels(self.levels):
            raise ValueError("Compaction levels must be unique and in supported order.")
        if not _matches_checkpoint_level(self.levels, self.checkpoint):
            raise ValueError("L3 requires an extractive checkpoint and L4 requires a handoff checkpoint.")
        if bool(self.replacements) != any(level in ("L1", "L2") for level in self.levels):
            raise ValueError("L1 and L2 require text replacements.")
        if self.after_tokens >= self.before_tokens:
            raise ValueError("A compaction commit must reduce the estimated input tokens.")
        return self


EventPayload = (
    SessionCreatedPayload | UserMessagePayload | ModelRequestStartedPayload
    | AssistantMessagePayload | ToolResultPayload | TurnFinishedPayload | CompactionCommittedPayload
)
_PAYLOAD_TYPES = {
    "session_created": SessionCreatedPayload,
    "user_message": UserMessagePayload,
    "model_request_started": ModelRequestStartedPayload,
    "assistant_message": AssistantMessagePayload,
    "tool_result": ToolResultPayload,
    "turn_finished": TurnFinishedPayload,
    "compaction_committed": CompactionCommittedPayload,
}


class SessionEvent(ProtocolModel):
    v: Literal[1] = 1
    session_id: UUID
    event_id: UUID
    seq: int = Field(ge=1)
    ts: datetime
    type: EventType
    payload: EventPayload

    @model_validator(mode="after")
    def match_payload_type(self) -> "SessionEvent":
        if not isinstance(self.payload, _PAYLOAD_TYPES[self.type]):
            raise ValueError("Event type does not match its payload.")
        if self.type == "session_created" and self.seq != 1:
            raise ValueError("The session creation event must have sequence 1.")
        return self


ConversationHistory = tuple[SessionEvent, ...]


class HistoryError(ValueError):
    pass


def check_event_position(event: SessionEvent, session_id: UUID, seq: int, known_ids: set[UUID]) -> None:
    """Check the envelope of an event in a complete session log."""
    if event.session_id != session_id:
        raise HistoryError("Event session ID does not match the log.")
    if event.seq != seq:
        raise HistoryError(f"Expected sequence {seq}, got {event.seq}.")
    if seq == 1 and event.type != "session_created":
        raise HistoryError("The log must start with session_created.")
    if event.event_id in known_ids:
        raise HistoryError("Duplicate event ID.")


class _ToolResultSource(NamedTuple):
    batch_seq: int
    call: ToolCall
    result: ToolResultPayload


@dataclass(frozen=True, slots=True)
class ArchiveBounds:
    start_seq: int
    covered_through_seq: int | None
    protected_from_seq: int | None
    protected_batches: tuple[int, ...] = ()
    stop_reason: str | None = None

    @property
    def covered_events(self) -> int:
        return 0 if self.covered_through_seq is None else self.covered_through_seq - self.start_seq + 1


class _TurnHistory:
    """Check turn and tool transitions without rescanning the accepted prefix."""

    def __init__(self) -> None:
        self.active_turn: UUID | None = None
        self.pending: deque[ToolCall] = deque()
        self._turn_ids: set[UUID] = set()
        self._assistant_seqs: set[int] = set()
        self._call_ids: set[UUID] = set()
        self._result_ids: set[UUID] = set()
        self.consumed_call_ids: set[UUID] = set()
        self._request_ids: set[UUID] = set()
        self._reply_request_ids: set[UUID] = set()
        self._ordinary_texts: dict[int, str] = {}
        self._recent_blocks: deque[int] = deque(maxlen=2)
        self._complete_blocks: list[tuple[int, int]] = []
        self._latest_user_seq: int | None = None
        self._batch_seq: int | None = None
        self._tool_batches: dict[int, list[_ToolResultSource]] = {}
        self._tool_sources: dict[int, _ToolResultSource] = {}
        self.projection_replacements: dict[int, TextReplacement] = {}
        self.checkpoint: Checkpoint | None = None
        self.latest_request: ModelRequestStartedPayload | None = None
        self.retried_transient = False
        self.retried_overflow = False

    def apply(self, event: SessionEvent, *, first: bool, history: Sequence[SessionEvent]) -> None:
        payload = event.payload
        if isinstance(payload, SessionCreatedPayload):
            if not first:
                raise HistoryError("Session creation must be the first event.")
            return
        if isinstance(payload, UserMessagePayload):
            if self.active_turn is not None or payload.turn_id in self._turn_ids:
                raise HistoryError("Previous turn must be closed before a new user message.")
            self.active_turn = payload.turn_id
            self._turn_ids.add(payload.turn_id)
            self._assistant_seqs.clear()
            self._latest_user_seq = event.seq
            self._ordinary_texts[event.seq] = payload.content
            self._record_block(event.seq, event.seq)
            self.latest_request = None
            self.retried_transient = False
            self.retried_overflow = False
            return

        if isinstance(payload, CompactionCommittedPayload):
            payload = CompactionCommittedPayload.model_validate(payload.model_dump(warnings=False))
            if payload.base_seq != event.seq - 1:
                raise HistoryError("The compaction base does not match the previous event.")
            self.validate_replacements(payload.replacements, payload.levels, payload.checkpoint, history)
            if payload.checkpoint is not None:
                self.checkpoint = payload.checkpoint
                self.projection_replacements = {
                    seq: item for seq, item in self.projection_replacements.items() if seq > self.covered_through_seq
                }
            self.projection_replacements.update((item.target_seq, item) for item in payload.replacements)
            return

        if payload.turn_id != self.active_turn:
            raise HistoryError("Event does not belong to the current turn.")
        if isinstance(payload, ToolResultPayload):
            if not self.pending or self.pending[0].call_id != payload.call_id:
                raise HistoryError("Tool result must settle the next accepted call.")
            assert self._batch_seq is not None
            call = self.pending.popleft()
            source = _ToolResultSource(self._batch_seq, call, payload)
            self._tool_sources[event.seq] = source
            self._tool_batches[self._batch_seq].append(source)
            self._result_ids.add(payload.call_id)
            if not self.pending:
                self._record_block(self._batch_seq, event.seq)
                self._batch_seq = None
            return
        if self.pending:
            raise HistoryError("All accepted calls must be settled before the next message or turn end.")

        if isinstance(payload, ModelRequestStartedPayload):
            self._record_request(payload, history[event.seq - 2].payload)
            return

        if isinstance(payload, AssistantMessagePayload):
            if payload.request_id in self._reply_request_ids:
                raise HistoryError("The model request already has an assistant response.")
            if self.latest_request is not None:
                if payload.request_id != self.latest_request.request_id:
                    raise HistoryError("Assistant reply does not match the latest registered request.")
            elif payload.request_id in self._request_ids:
                raise HistoryError("Duplicate model request ID.")
            if not self._result_ids.issuperset(payload.consumed_call_ids):
                raise HistoryError("Assistant consumption refers to a missing tool result.")
            local_ids = {call.call_id for call in payload.tool_calls}
            provider_ids = {call.provider_call_id for call in payload.tool_calls}
            if (
                len(local_ids) != len(payload.tool_calls)
                or len(provider_ids) != len(payload.tool_calls)
                or not self._call_ids.isdisjoint(local_ids)
            ):
                raise HistoryError("Duplicate identity in an accepted tool batch.")
            self._reply_request_ids.add(payload.request_id)
            self._request_ids.add(payload.request_id)
            self._assistant_seqs.add(event.seq)
            self._call_ids.update(local_ids)
            self.consumed_call_ids.update(payload.consumed_call_ids)
            self.pending.extend(payload.tool_calls)
            if self.pending:
                self._batch_seq = event.seq
                self._tool_batches[event.seq] = []
            elif payload.content:
                self._record_block(event.seq, event.seq)
                if not payload.rejected_tool_calls:
                    self._ordinary_texts[event.seq] = payload.content
        else:
            if payload.assistant_seq is not None and payload.assistant_seq not in self._assistant_seqs:
                raise HistoryError("Turn end refers to an assistant message outside the turn.")
            self.active_turn = None

    def _record_block(self, start_seq: int, end_seq: int) -> None:
        self._recent_blocks.append(start_seq)
        self._complete_blocks.append((start_seq, end_seq))

    @property
    def covered_through_seq(self) -> int:
        return self.checkpoint.covered_through_seq if self.checkpoint is not None else 0

    def _require_compaction_boundary(self) -> None:
        if self.active_turn is None or self.pending:
            raise HistoryError("Compaction requires an open turn without an unsettled tool batch.")

    def _l1_target_is_available(self, seq: int) -> bool:
        return (
            seq > self.covered_through_seq and seq != self._latest_user_seq
            and seq not in self._recent_blocks and seq not in self.projection_replacements
        )

    def l1_sources(self) -> dict[int, str]:
        self._require_compaction_boundary()
        return {
            seq: content for seq, content in self._ordinary_texts.items() if self._l1_target_is_available(seq)
        }

    def _l2_batch_is_available(self, seq: int) -> bool:
        # Preserve the entire transaction if any sibling lacks durable evidence.
        return seq not in self._recent_blocks and all(
            source.call.call_id in self.consumed_call_ids
            and source.result.status in ("ok", "error")
            and source.result.details.get("execution_outcome") not in ("unknown", "not_started")
            for source in self._tool_batches[seq]
        )

    def l2_sources(self) -> dict[int, _ToolResultSource]:
        self._require_compaction_boundary()
        eligible_batches = {seq for seq in self._tool_batches if self._l2_batch_is_available(seq)}
        return {
            seq: source for seq, source in self._tool_sources.items()
            if seq > self.covered_through_seq and seq not in self.projection_replacements
            and source.batch_seq in eligible_batches
        }

    def l3_bounds(self) -> ArchiveBounds:
        start = self.covered_through_seq + 1
        protected_batches = tuple(
            seq for seq in self._tool_batches
            if seq >= start and (seq == self._batch_seq or not self._l2_batch_is_available(seq))
        )
        if self.pending:
            return ArchiveBounds(start, None, self._batch_seq, protected_batches, "pending_tool_calls")
        if self.active_turn is None:
            return ArchiveBounds(start, None, None, protected_batches, "no_open_turn")
        assert self._latest_user_seq is not None
        protected_from = min(self._latest_user_seq, *self._recent_blocks, *protected_batches)
        covered = next((
            end for head, end in reversed(self._complete_blocks) if start <= head and end < protected_from
        ), None)
        reason = None
        if covered is None:
            reason = "protected_tool_transaction" if protected_from in protected_batches else "no_complete_prefix"
        return ArchiveBounds(start, covered, protected_from, protected_batches, reason)

    def _validate_checkpoint(self, checkpoint: Checkpoint, history: Sequence[SessionEvent]) -> None:
        bounds = self.l3_bounds()
        end = checkpoint.covered_through_seq
        if (
            bounds.protected_from_seq is None or not bounds.start_seq <= end < bounds.protected_from_seq
            or end > len(history) or history[end - 1].seq != end
            or not any(bounds.start_seq <= head and tail <= end for head, tail in self._complete_blocks)
            or any(head <= end < tail for head, tail in self._complete_blocks)
        ):
            raise HistoryError("Checkpoint must extend the legal prefix without crossing protected or partial blocks.")
        previous_items = checkpoint_evidence(self.checkpoint)
        if isinstance(checkpoint, HandoffCheckpoint):
            assert self._latest_user_seq is not None
            goal = self._ordinary_texts[self._latest_user_seq]
            valid = l4.is_grounded(
                checkpoint.summary, history[:end], end, goal, previous_items, self.covered_through_seq,
                preserve_previous=isinstance(self.checkpoint, HandoffCheckpoint),
            )
        else:
            valid = l3.is_extractive_summary(
                checkpoint.summary, history[:end], end, previous_items, self.covered_through_seq,
            )
            if isinstance(self.checkpoint, HandoffCheckpoint):
                valid = valid and set(previous_items).issubset(checkpoint.summary)
        if not valid:
            raise HistoryError("Checkpoint summary does not match its original evidence or inherited entries.")

    def validate_replacements(
        self, replacements: tuple[TextReplacement, ...], levels: CompactionLevels,
        checkpoint: Checkpoint | None = None, history: Sequence[SessionEvent] = (),
    ) -> None:
        self._require_compaction_boundary()
        if not _valid_compaction_levels(levels):
            raise HistoryError("Compaction levels must be unique and in supported order.")
        if not _matches_checkpoint_level(levels, checkpoint):
            raise HistoryError("L3 requires an extractive checkpoint and L4 requires a handoff checkpoint.")
        targets: set[int] = set()
        used_levels: set[CompactionLevel] = set()
        covered = self.covered_through_seq
        if checkpoint is not None:
            self._validate_checkpoint(checkpoint, history)
            covered = checkpoint.covered_through_seq
            used_levels.update(_checkpoint_level(checkpoint))
        for item in replacements:
            seq = item.target_seq
            if seq <= covered:
                raise HistoryError("A compaction replacement cannot target archived history.")
            if seq in targets or seq in self.projection_replacements:
                raise HistoryError("Compaction targets must be unique and not already replaced.")
            if "L1" in levels and seq in self._ordinary_texts and self._l1_target_is_available(seq):
                if not l1.is_display_edge_trim(self._ordinary_texts[seq], item.content):
                    raise HistoryError("The L1 replacement changes more than the allowed display edges.")
                used_levels.add("L1")
            elif "L2" in levels and seq in self._tool_sources:
                source = self._tool_sources[seq]
                if not self._l2_batch_is_available(source.batch_seq):
                    raise HistoryError("The L2 transaction is protected or lacks durable consumption evidence.")
                if not l2.is_tool_result_excerpt(seq, source.call, source.result, item.content):
                    raise HistoryError("The L2 replacement does not match the original tool evidence.")
                used_levels.add("L2")
            else:
                raise HistoryError("The compaction target is protected or invalid for the selected levels.")
            targets.add(seq)
        if used_levels != set(levels):
            raise HistoryError("Each compaction level must contribute a replacement or checkpoint.")

    def _record_request(self, payload: ModelRequestStartedPayload, preceding: EventPayload) -> None:
        previous = self.latest_request
        if previous is None and self._assistant_seqs:
            raise HistoryError("Request tracking cannot begin after an unregistered reply.")
        if payload.attempt != (1 if previous is None else previous.attempt + 1):
            raise HistoryError("Model attempts must be sequential within a turn.")
        if (payload.trigger == "initial") != (previous is None):
            raise HistoryError("Only the first model attempt in a turn uses the initial trigger.")
        if payload.request_id in self._request_ids:
            raise HistoryError("Duplicate model request ID.")
        if payload.trigger == "retry_transient" and (
            self.retried_transient or previous is None or previous.request_id in self._reply_request_ids
        ):
            raise HistoryError("A transient retry requires an unanswered attempt and unused retry budget.")
        if payload.trigger == "retry_overflow" and (
            self.retried_overflow or previous is None or previous.request_id in self._reply_request_ids
            or not isinstance(preceding, CompactionCommittedPayload) or preceding.trigger != "overflow"
        ):
            raise HistoryError("An overflow retry requires an unanswered attempt, an overflow commit and unused retry budget.")
        self._request_ids.add(payload.request_id)
        self.latest_request = payload
        if payload.trigger == "retry_transient":
            self.retried_transient = True
        if payload.trigger == "retry_overflow":
            self.retried_overflow = True


def inspect_projection_history(history: ConversationHistory) -> _TurnHistory:
    state = _TurnHistory()
    for position, event in enumerate(history):
        state.apply(event, first=position == 0, history=history)
    return state


def inspect_history(history: ConversationHistory) -> tuple[UUID | None, tuple[ToolCall, ...]]:
    """Check event order and return an open turn and any unsettled calls."""
    state = inspect_projection_history(history)
    return state.active_turn, tuple(state.pending)
