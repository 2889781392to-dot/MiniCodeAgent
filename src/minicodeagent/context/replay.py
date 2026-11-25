"""Fold session events and rebuild a read-only view of a persisted session."""

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from ..protocol import ToolCall
from .builder import ContextBuilder, ContextProjection
from .events import (
    AssistantMessagePayload,
    Checkpoint,
    ConversationHistory,
    HistoryError,
    ModelRequestStartedPayload,
    SessionCreatedPayload,
    SessionEvent,
    TextReplacement,
    ToolResultPayload,
    TurnFinishedPayload,
    UserMessagePayload,
    _TurnHistory,
    check_event_position,
)

if TYPE_CHECKING:
    from .store import JsonlEventStore, SessionLogLease


@dataclass(frozen=True, slots=True)
class TurnState:
    turn_id: UUID
    user_seq: int
    tool_batches: int = 0
    pending_calls: tuple[ToolCall, ...] = ()
    latest_reply: SessionEvent | None = None
    finished: TurnFinishedPayload | None = None
    latest_request: ModelRequestStartedPayload | None = None
    retried_transient: bool = False
    retried_overflow: bool = False

    @property
    def model_attempts(self) -> int:
        return self.latest_request.attempt if self.latest_request is not None else 0


@dataclass(slots=True)
class SessionState:
    """Derived state for the events in one session log."""

    _history: list[SessionEvent] = field(default_factory=list, init=False, repr=False)
    _event_ids: set[UUID] = field(default_factory=set, init=False, repr=False, compare=False)
    _turns: _TurnHistory = field(default_factory=_TurnHistory, init=False, repr=False, compare=False)
    _turn: TurnState | None = field(default=None, init=False, repr=False)

    @property
    def session_id(self) -> UUID | None:
        return self._history[0].session_id if self._history else None

    @property
    def metadata(self) -> SessionCreatedPayload | None:
        if not self._history:
            return None
        payload = self._history[0].payload
        assert isinstance(payload, SessionCreatedPayload)
        return payload

    @property
    def history(self) -> ConversationHistory:
        return tuple(event.model_copy(deep=True) for event in self._history)

    @property
    def last_seq(self) -> int:
        return len(self._history)

    @property
    def next_seq(self) -> int:
        return self.last_seq + 1

    @property
    def current_turn(self) -> TurnState | None:
        if self._turn is not None and self._turns.pending:
            return replace(self._turn, pending_calls=tuple(self._turns.pending))
        return self._turn

    @property
    def consumed_call_ids(self) -> frozenset[UUID]:
        return frozenset(self._turns.consumed_call_ids)

    @property
    def projection_replacements(self) -> tuple[TextReplacement, ...]:
        """The validated text overlays, ordered by their original event sequence."""
        return tuple(item for _, item in sorted(self._turns.projection_replacements.items()))

    @property
    def checkpoint(self) -> Checkpoint | None:
        return self._turns.checkpoint


class SessionReducer:
    def apply(self, state: SessionState, event: SessionEvent) -> None:
        """Validate and apply one event, leaving state untouched on validation errors."""
        check_event_position(event, state.session_id or event.session_id, state.next_seq, state._event_ids)
        event = event.model_copy(deep=True)
        state._turns.apply(event, first=state.last_seq == 0, history=state._history)

        payload = event.payload
        if isinstance(payload, UserMessagePayload):
            state._turn = TurnState(turn_id=payload.turn_id, user_seq=event.seq)
        elif isinstance(payload, ModelRequestStartedPayload):
            assert state._turn is not None
            state._turn = replace(
                state._turn, latest_request=payload, retried_transient=state._turns.retried_transient,
                retried_overflow=state._turns.retried_overflow,
            )
        elif isinstance(payload, AssistantMessagePayload):
            assert state._turn is not None
            state._turn = replace(
                state._turn,
                latest_reply=event,
                tool_batches=state._turn.tool_batches + bool(payload.tool_calls),
            )
        elif isinstance(payload, TurnFinishedPayload):
            assert state._turn is not None
            state._turn = replace(state._turn, finished=payload)

        state._history.append(event)
        state._event_ids.add(event.event_id)

    def replay(self, events: Iterable[SessionEvent]) -> SessionState:
        """Fold an ordered iterable of events into a fresh state."""
        state = SessionState()
        for event in events:
            self.apply(state, event)
        return state


class EmptySessionLogError(HistoryError):
    pass


@dataclass(frozen=True, slots=True)
class ReplayDiagnostic:
    code: Literal["pending_tool_results", "unfinished_turn", "projection_failed"]
    message: str


@dataclass(frozen=True, slots=True)
class SessionReplayResult:
    state: SessionState
    projection: ContextProjection | None
    unknown_call_ids: frozenset[UUID]
    diagnostics: tuple[ReplayDiagnostic, ...]

    @property
    def pending_calls(self) -> tuple[ToolCall, ...]:
        turn = self.state.current_turn
        return turn.pending_calls if turn is not None else ()

    @property
    def consumed_call_ids(self) -> frozenset[UUID]:
        return self.state.consumed_call_ids

    @property
    def last_seq(self) -> int:
        return self.state.last_seq

    @property
    def next_seq(self) -> int:
        return self.state.next_seq


class SessionReplay:
    def __init__(
        self, store: "JsonlEventStore | SessionLogLease", builder: ContextBuilder | None = None,
    ) -> None:
        self._store = store
        self._builder = builder if builder is not None else ContextBuilder()

    def replay(self, session_id: UUID, system_prompt: str) -> SessionReplayResult:
        state = SessionReducer().replay(self._store.iter_events(session_id, require_log=True))
        if state.last_seq == 0:
            raise EmptySessionLogError(f"Session {session_id} has an empty event log.")

        history = state.history
        turn = state.current_turn
        pending = turn.pending_calls if turn is not None else ()
        unknown_ids = {call.call_id for call in pending}
        for event in history:
            payload = event.payload
            if isinstance(payload, ToolResultPayload) and payload.details.get("execution_outcome") == "unknown":
                unknown_ids.add(payload.call_id)

        diagnostics = []
        if turn is not None and turn.finished is None:
            diagnostics.append(ReplayDiagnostic("unfinished_turn", "The latest turn is still open."))

        projection = None
        if pending:
            diagnostics.append(ReplayDiagnostic(
                "pending_tool_results", "Tool results are missing. Execution outcomes are unknown.",
            ))
        else:
            try:
                projection = self._builder.project(history, system_prompt)
            except HistoryError as exc:
                diagnostics.append(ReplayDiagnostic("projection_failed", str(exc)))

        return SessionReplayResult(state, projection, frozenset(unknown_ids), tuple(diagnostics))
