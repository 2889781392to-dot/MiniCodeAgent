"""Open, recover, and branch writable sessions under their log locks."""

from collections.abc import Callable
import logging
from pathlib import Path
from uuid import UUID, uuid4

from ..context.events import (
    AssistantMessagePayload, SessionCreatedPayload, SessionEvent, ToolResultPayload,
    TurnFinishedPayload,
)
from ..context.replay import SessionReducer, SessionReplay, SessionReplayResult, SessionState
from ..context.store import EventStoreError, JsonlEventSink, JsonlEventStore, SessionLogLease
from ..context.writer import SessionEventWriter
from ..protocol import ToolCall
from ..workspace import Workspace, WorkspaceError
from .catalog import SessionCatalog, SessionCatalogError, SessionRecord

_logger = logging.getLogger(__name__)


class ResumeError(ValueError):
    pass


class SessionHandle:
    def __init__(
        self, store: SessionLogLease, workspace: Workspace, title: str,
        replay: SessionReplay, system_prompt: str, catalog: SessionCatalog,
        on_event: Callable[[SessionEvent], None] | None = None,
    ) -> None:
        self.store = store
        self.workspace = workspace
        self.title = title
        self.writer = SessionEventWriter(
            session_id=store.session_id, sink=JsonlEventSink(store, store.session_id),
            on_event=on_event,
        )
        self.recovered_reply: SessionEvent | None = None
        self._replay = replay
        self._system_prompt = system_prompt
        self._catalog = catalog

    @property
    def session_id(self) -> UUID:
        return self.store.session_id

    @property
    def closed(self) -> bool:
        return self.store.closed

    @property
    def replay_result(self) -> SessionReplayResult:
        return self._replay.replay(self.session_id, self._system_prompt)

    @property
    def state(self) -> SessionState:
        return self.replay_result.state

    def __enter__(self) -> "SessionHandle":
        if self.closed:
            raise EventStoreError("Session handle is closed.")
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.store.close()
        try:
            self._catalog.refresh(self.session_id)
        except SessionCatalogError as exc:
            _logger.warning("Session %s closed, but its index could not be refreshed: %s", self.session_id, exc)


class SessionService:
    def __init__(
        self, store: JsonlEventStore, *, system_prompt: str = "",
        on_event: Callable[[SessionEvent], None] | None = None,
    ) -> None:
        self._store = store
        self._system_prompt = system_prompt
        self._on_event = on_event
        self.catalog = SessionCatalog(store)

    def create(self, workspace: str | Path, title: str | None = None) -> SessionHandle:
        bound = Workspace(Path(workspace))
        title = (bound.root.name or str(bound.root)) if title is None else title
        session_id = uuid4()
        lease = self._store.lock_session(session_id)
        try:
            event = SessionEventWriter(session_id=session_id).append(
                "session_created", SessionCreatedPayload(title=title, workspace=str(bound.root)),
            )
            handle = SessionHandle(
                lease, bound, title, SessionReplay(lease), self._system_prompt, self.catalog, self._on_event,
            )
            lease.publish(session_id, (event,))
            return handle
        except BaseException:
            lease.close()
            raise

    new = create

    def list(self) -> list[SessionRecord]:
        return self.catalog.list()

    def open(self, session_id: UUID, *, workspace: str | Path | None = None) -> SessionHandle:
        lease = self._store.lock_session(session_id)
        try:
            return self._open_locked(lease, workspace)
        except BaseException:
            lease.close()
            raise

    def _open_locked(self, lease: SessionLogLease, workspace: str | Path | None) -> SessionHandle:
        replay = SessionReplay(lease)
        metadata = replay.replay(lease.session_id, self._system_prompt).state.metadata
        assert metadata is not None
        bound = _bind_workspace(metadata.workspace, workspace)
        return SessionHandle(
            lease, bound, metadata.title, replay, self._system_prompt, self.catalog, self._on_event,
        )

    def resume(
        self, session_id: UUID, *, workspace: str | Path | None = None, continue_turn: bool = False,
        confirm_pending: Callable[[tuple[ToolCall, ...]], bool] | None = None,
    ) -> SessionHandle:
        handle = self.open(session_id, workspace=workspace)
        try:
            state = handle.state
            turn = state.current_turn
            if turn is None or turn.finished is not None:
                if continue_turn:
                    raise ResumeError("There is no unfinished turn to continue.")
                return handle
            reply = turn.latest_reply
            restored = _recorded_terminal(state)
            if restored is not None:
                handle.writer.append("turn_finished", restored)
                handle.recovered_reply = reply
                return handle
            if continue_turn:
                if turn.latest_request is None:
                    raise ResumeError("The remaining model request budget is unknown for this legacy turn.")
                if reply is not None and reply.seq == state.last_seq:
                    payload = reply.payload
                    assert isinstance(payload, AssistantMessagePayload)
                    if not payload.tool_calls:
                        raise ResumeError("The saved reply has no recoverable terminal. Resume without --continue.")

            pending = turn.pending_calls
            if pending:
                if confirm_pending is None or not confirm_pending(pending):
                    call_ids = ", ".join(str(call.call_id) for call in pending)
                    raise ResumeError(f"Confirmation required for pending tool calls with unknown outcomes ({call_ids}).")
                for call in pending:
                    handle.writer.append("tool_result", ToolResultPayload(
                        turn_id=turn.turn_id, call_id=call.call_id, status="interrupted",
                        output="Recovery found no recorded result. Execution outcome is unknown. Verify effects before retrying.",
                        details={"execution_outcome": "unknown"},
                    ))
            if not continue_turn:
                handle.writer.append("turn_finished", TurnFinishedPayload(
                    turn_id=turn.turn_id, status="interrupted", reason="interrupted",
                ))
            return handle
        except BaseException:
            handle.close()
            raise

    def fork(
        self, source_session_id: UUID, *, workspace: str | Path | None = None, title: str | None = None,
    ) -> SessionHandle:
        with self._store.lock_session(source_session_id) as source:
            state = SessionReplay(source).replay(source_session_id, self._system_prompt).state
        metadata = state.metadata
        assert metadata is not None
        bound = _bind_workspace(metadata.workspace, workspace)
        session_id = uuid4()
        lease = self._store.lock_session(session_id)
        try:
            header = SessionEventWriter(session_id=session_id).append(
                "session_created", SessionCreatedPayload(
                    title=f"Fork of {metadata.title}" if title is None else title,
                    workspace=str(bound.root), parent_session_id=source_session_id,
                    parent_through_seq=state.last_seq,
                ),
            )
            events = (header, *(event.model_copy(update={
                "session_id": session_id, "event_id": uuid4(),
            }) for event in state.history[1:]))
            SessionReducer().replay(events)
            lease.publish(session_id, events)
            try:
                return self._open_locked(lease, bound.root)
            except (EventStoreError, ValueError) as exc:
                raise EventStoreError(f"Fork {session_id} was published. Reopening failed with {exc}") from exc
        except BaseException:
            lease.close()
            raise

    @staticmethod
    def close(handle: SessionHandle) -> None:
        handle.close()


def _bind_workspace(saved: str, override: str | Path | None) -> Workspace:
    if override is not None:
        return Workspace(Path(override))
    saved_root = Path(saved)
    if not saved_root.is_absolute():
        raise WorkspaceError(f"Saved workspace {saved!r} is not absolute on this host.")
    try:
        bound = Workspace(saved_root)
    except WorkspaceError as exc:
        raise WorkspaceError(f"Saved workspace {saved!r} is unavailable.") from exc
    if bound.root != saved_root:
        raise WorkspaceError(f"Saved workspace {saved!r} resolves to a different directory.")
    return bound


def _recorded_terminal(state: SessionState) -> TurnFinishedPayload | None:
    turn = state.current_turn
    assert turn is not None
    event = turn.latest_reply
    if turn.pending_calls or event is None or event.seq != state.last_seq:
        return None
    payload = event.payload
    assert isinstance(payload, AssistantMessagePayload)
    if payload.tool_calls or payload.rejected_tool_calls:
        return None
    if payload.finish_reason == "stop" and payload.content.strip():
        return TurnFinishedPayload(
            turn_id=turn.turn_id, status="completed", reason="model_stop", assistant_seq=event.seq,
        )
    if payload.finish_reason == "length":
        return TurnFinishedPayload(
            turn_id=turn.turn_id, status="limited", reason="output_length", assistant_seq=event.seq,
        )
    if payload.finish_reason not in {"tool_calls", "content_filter"}:
        return TurnFinishedPayload(
            turn_id=turn.turn_id, status="failed", reason="unexpected_finish_reason", assistant_seq=event.seq,
        )
    return None
