"""Build and validate events before handing them to a session's event sink."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from .events import ConversationHistory, EventPayload, EventType, SessionEvent, inspect_history


class EventSink(Protocol):
    @property
    def history(self) -> ConversationHistory: ...

    def append(self, event: SessionEvent) -> None: ...


class MemoryEventSink:
    def __init__(self) -> None:
        self._events: list[SessionEvent] = []

    @property
    def history(self) -> ConversationHistory:
        return tuple(event.model_copy(deep=True) for event in self._events)

    def append(self, event: SessionEvent) -> None:
        self._events.append(event.model_copy(deep=True))


class EventWriteError(RuntimeError):
    pass


class SessionEventWriter:
    def __init__(
        self,
        *,
        session_id: UUID | None = None,
        sink: EventSink | None = None,
        on_event: Callable[[SessionEvent], None] | None = None,
    ) -> None:
        self.session_id = session_id or uuid4()
        self._sink = sink if sink is not None else MemoryEventSink()
        self._on_event = on_event

    @property
    def history(self) -> ConversationHistory:
        try:
            return self._sink.history
        except Exception as exc:
            raise EventWriteError("Could not read event history.") from exc

    def append(self, event_type: EventType, payload: EventPayload) -> SessionEvent:
        history = self.history
        try:
            event = SessionEvent(
                session_id=self.session_id,
                event_id=uuid4(),
                seq=history[-1].seq + 1 if history else 1,
                ts=datetime.now(UTC),
                type=event_type,
                payload=payload,
            )
            inspect_history((*history, event))
            self._sink.append(event)
        except Exception as exc:
            raise EventWriteError(f"Could not append {event_type}.") from exc

        if self._on_event is not None:
            try:
                self._on_event(event.model_copy(deep=True))
            except Exception:
                # The subscriber only displays committed events.
                pass
        return event
