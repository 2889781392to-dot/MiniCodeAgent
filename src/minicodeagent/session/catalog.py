"""Rebuild the session catalog from the committed event logs."""

from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import stat
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import UUID

from pydantic import Field, ValidationError, field_validator

from ..context.events import HistoryError
from ..context.replay import SessionReducer
from ..context.store import EventStoreBusyError, EventStoreError, JsonlEventStore, file_lock
from ..protocol import ProtocolModel, TurnStatus


class SessionCatalogError(RuntimeError):
    """The derived catalog could not be read or updated."""


class SessionRecord(ProtocolModel):
    id: UUID | None
    log_name: str
    status: Literal["valid", "corrupt", "busy"]
    title: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_seq: int | None = Field(default=None, ge=1)
    turn_status: TurnStatus | Literal["open"] | None = None
    parent_session_id: UUID | None = None
    parent_through_seq: int | None = Field(default=None, ge=1)
    size_bytes: int | None = Field(default=None, ge=0)
    mtime_ns: int | None = None
    error: str | None = None

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() != timedelta(0):
            raise ValueError("Catalog timestamps must be UTC.")
        return value


class _Index(ProtocolModel):
    version: int = Field(ge=1, le=1)
    sessions: dict[str, SessionRecord]


class SessionCatalog:
    def __init__(self, store: JsonlEventStore) -> None:
        self._store = store
        self.root = store.root
        self._index_path = self.root / "session_index.json"
        self._lock_path = self.root / "locks" / "session_index.lock"

    def list(self) -> list[SessionRecord]:
        try:
            sessions_dir = self.root / "sessions"
            if not sessions_dir.exists() and not self._index_path.exists():
                return []
            with file_lock(self._lock_path, "session index"):
                previous = self._read_index()
                cached = previous.sessions if previous is not None else {}
                index: dict[str, SessionRecord] = {}
                records = []
                try:
                    paths = tuple(sessions_dir.iterdir())
                except FileNotFoundError:
                    paths = ()
                for path in paths:
                    if path.suffix.casefold() != ".jsonl":
                        continue
                    summary = cached.get(path.name)
                    record = self._read_record(path, summary)
                    records.append(record)
                    if record.status == "valid":
                        index[path.name] = record
                    elif record.status == "busy" and summary is not None:
                        index[path.name] = summary
                current = _Index(version=1, sessions=index)
                if current != previous:
                    self._write_index(current)
        except (EventStoreError, OSError) as exc:
            raise SessionCatalogError(f"Could not refresh session index: {exc}") from exc
        oldest = datetime.min.replace(tzinfo=UTC)
        return sorted(
            records,
            key=lambda record: (record.updated_at or oldest, str(record.id) if record.id else record.log_name),
            reverse=True,
        )

    def refresh(self, session_id: UUID) -> None:
        """Reconcile the derived index after a session changes."""
        if not isinstance(session_id, UUID):
            raise SessionCatalogError("Session ID must be a UUID.")
        self.list()

    def _read_index(self) -> _Index | None:
        try:
            index = _Index.model_validate_json(self._index_path.read_bytes())
        except (FileNotFoundError, ValidationError):
            return None
        for name, record in index.sessions.items():
            if (
                record.status != "valid"
                or record.id is None
                or name != record.log_name
                or name != f"{record.id}.jsonl"
                or any(value is None for value in (
                    record.title, record.workspace, record.created_at, record.updated_at,
                    record.last_seq, record.size_bytes, record.mtime_ns,
                ))
            ):
                return None
        return index

    def _read_record(self, path: Path, cached: SessionRecord | None) -> SessionRecord:
        try:
            session_id = UUID(path.stem)
            if path.name != f"{session_id}.jsonl":
                raise ValueError("Noncanonical session log name.")
        except ValueError:
            return SessionRecord(
                id=None, log_name=path.name, status="corrupt", error="Invalid session log filename.",
            )
        try:
            with self._store.lock_session(session_id) as lease:
                signature = path.lstat()
                if not stat.S_ISREG(signature.st_mode) or (
                    getattr(signature, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    raise HistoryError("Expected a regular session log file.")
                if cached is not None and (
                    cached.size_bytes, cached.mtime_ns
                ) == (signature.st_size, signature.st_mtime_ns):
                    return cached
                events = tuple(lease.iter_events(session_id, require_log=True))
                state = SessionReducer().replay(events)
                metadata = state.metadata
                if metadata is None:
                    raise HistoryError("The session log is empty.")
                turn = state.current_turn
                turn_status = None if turn is None else (
                    "open" if turn.finished is None else turn.finished.status
                )
                return SessionRecord(
                    id=session_id, log_name=path.name, status="valid",
                    title=metadata.title, workspace=metadata.workspace,
                    created_at=events[0].ts, updated_at=events[-1].ts, last_seq=state.last_seq,
                    turn_status=turn_status, parent_session_id=metadata.parent_session_id,
                    parent_through_seq=metadata.parent_through_seq,
                    size_bytes=signature.st_size, mtime_ns=signature.st_mtime_ns,
                )
        except EventStoreBusyError as exc:
            if cached is not None:
                return cached.model_copy(update={"status": "busy", "error": str(exc)})
            return SessionRecord(id=session_id, log_name=path.name, status="busy", error=str(exc))
        except (EventStoreError, HistoryError, OSError) as exc:
            return SessionRecord(id=session_id, log_name=path.name, status="corrupt", error=str(exc))

    def _write_index(self, index: _Index) -> None:
        staged = None
        try:
            with NamedTemporaryFile(
                mode="wb", dir=self.root, prefix=".session-index-", suffix=".tmp", delete=False,
            ) as file:
                staged = Path(file.name)
                data = index.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
                if file.write(data) != len(data):
                    raise OSError("Could not finish writing the temporary session index.")
                file.flush()
                os.fsync(file.fileno())
            os.replace(staged, self._index_path)
            staged = None
        finally:
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    # Leave a failed temporary write out of the catalog.
                    pass
