"""Persist validated session events as an append-only UTF-8 JSONL log."""

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
import errno
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import BinaryIO
from uuid import UUID

from pydantic import ValidationError

from .events import ConversationHistory, SessionEvent, check_event_position

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class EventStoreError(RuntimeError):
    pass


class EventStoreBusyError(EventStoreError):
    pass


class EventLogNotFoundError(EventStoreError):
    pass


class EventStoreCorruptError(EventStoreError):
    def __init__(self, session_id: UUID, line_number: int, last_valid_seq: int, reason: str) -> None:
        self.session_id = session_id
        self.line_number = line_number
        self.last_valid_seq = last_valid_seq
        super().__init__(
            f"Session {session_id}, line {line_number}, last valid seq {last_valid_seq}: {reason}"
        )


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("Duplicate JSON object field.")
        fields[key] = value
    return fields


def _reject_constant(_value: str) -> None:
    raise ValueError("Non-finite JSON number.")


def _parse_float(value: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError("Non-finite JSON number.")
    return number


def _decode(line: bytes) -> SessionEvent:
    # Check the wire object before Pydantic can fill defaults or coerce a literal.
    try:
        text = line.decode("utf-8")
        record = json.loads(
            text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("Invalid UTF-8 JSON.") from None
    if not isinstance(record, dict):
        raise ValueError("Expected a JSON event object.")
    if type(record.get("v")) is not int or record["v"] != 1:
        raise ValueError("Missing or unsupported event schema version.")
    try:
        timestamp = datetime.fromisoformat(record.get("ts"))
    except (TypeError, ValueError):
        raise ValueError("Expected an ISO-8601 UTC timestamp.") from None
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("Expected an ISO-8601 UTC timestamp.")
    payload = record.get("payload")
    if record.get("type") == "assistant_message" and isinstance(payload, dict):
        if "tool_calls" not in payload or "consumed_call_ids" not in payload:
            raise ValueError("Missing assistant tool call or consumption list.")
        for key in ("tool_calls", "rejected_tool_calls"):
            calls = payload.get(key)
            if isinstance(calls, list) and any(
                isinstance(call, dict) and "call_id" not in call for call in calls
            ):
                raise ValueError("Missing persisted tool call ID.")
    try:
        event = SessionEvent.model_validate_json(text)
    except ValidationError:
        raise ValueError("Invalid event schema.") from None
    return event


def _encode(event: SessionEvent) -> tuple[SessionEvent, bytes]:
    try:
        record = event.model_dump(mode="json", exclude_none=True, warnings="error")
        line = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        validated = _decode(line)
    except (ValueError, TypeError, RecursionError) as exc:
        raise EventStoreError("Could not serialize a valid session event.") from exc
    return validated, line


def _write_line(file: BinaryIO, line: bytes) -> None:
    if file.write(line) != len(line):
        raise OSError("Incomplete event log write.")


@contextmanager
def file_lock(lock_path: Path, label: str, *, create: bool = True) -> Iterator[None]:
    try:
        if create:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file = lock_path.open("a+b" if create else "rb")
        except FileNotFoundError:
            if create:
                raise
            # A copied log may have no sidecar. Writers create it before touching the log.
            read_error: EventStoreError | None = None
            try:
                yield
            except EventStoreError as exc:
                read_error = exc
            if lock_path.exists():
                raise EventStoreBusyError(f"{label} became active during read.") from read_error
            if read_error is not None:
                raise read_error
            return
        # Keep the sidecar in place. Closing the handle releases the OS lock.
        with file:
            try:
                if os.name == "nt":
                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise EventStoreBusyError(f"{label} is busy.") from exc
                raise
            yield
    except OSError as exc:
        raise EventStoreError(f"Could not use lock for {label}.") from exc


class JsonlEventStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).absolute()

    def lock_session(self, session_id: UUID) -> "SessionLogLease":
        return SessionLogLease(self, session_id)

    def append(self, event: SessionEvent) -> None:
        path = self._session_path(event.session_id)
        with self._session_lock(event.session_id):
            self._append_locked(path, event)

    def _append_locked(self, path: Path, event: SessionEvent) -> None:
        event, line = _encode(event)
        history = self._read(path, event.session_id)
        try:
            check_event_position(
                event, event.session_id, len(history) + 1, {item.event_id for item in history},
            )
        except ValueError as exc:
            raise EventStoreError(f"Cannot append to session {event.session_id}: {exc}") from exc
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as file:
                _write_line(file, line)
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            raise EventStoreError(
                f"Could not commit event to session {event.session_id}. Reread the log before retrying."
            ) from exc

    def iter_events(self, session_id: UUID, *, require_log: bool = False) -> Iterator[SessionEvent]:
        path = self._session_path(session_id)
        with self._session_lock(session_id, create=False):
            # The iterator consumes a checked snapshot after the lock is released.
            return iter(self._read(path, session_id, require_log=require_log))

    def publish(self, session_id: UUID, events: Iterable[SessionEvent]) -> None:
        """Sync a complete new log before making its name visible."""
        path = self._session_path(session_id)
        with self._session_lock(session_id):
            self._publish_locked(path, session_id, events)

    def _publish_locked(self, path: Path, session_id: UUID, events: Iterable[SessionEvent]) -> None:
        staged: Path | None = None
        try:
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise EventStoreError(f"Session {session_id} already has a log.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as file:
                staged = Path(file.name)
                known_ids: set[UUID] = set()
                for seq, event in enumerate(events, start=1):
                    event, line = _encode(event)
                    check_event_position(event, session_id, seq, known_ids)
                    known_ids.add(event.event_id)
                    _write_line(file, line)
                if not known_ids:
                    raise ValueError("A published log requires a session_created event.")
                file.flush()
                os.fsync(file.fileno())
            # Check the synced staging file before publishing its official name.
            if len(self._read(staged, session_id, require_log=True)) != len(known_ids):
                raise ValueError("Staged log has an unexpected record count.")
            os.replace(staged, path)
            staged = None
        except ValueError as exc:
            raise EventStoreError(f"Cannot publish session {session_id}: {exc}") from exc
        except OSError as exc:
            raise EventStoreError(f"Could not publish session {session_id}.") from exc
        finally:
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    # A failed cleanup must not mask the original publish error.
                    pass

    def _session_path(self, session_id: UUID) -> Path:
        if not isinstance(session_id, UUID):
            raise EventStoreError("Session ID must be a UUID.")
        return self.root / "sessions" / f"{session_id}.jsonl"

    @staticmethod
    def _read(path: Path, session_id: UUID, *, require_log: bool = False) -> list[SessionEvent]:
        try:
            file = path.open("rb")
        except FileNotFoundError as exc:
            if require_log:
                raise EventLogNotFoundError(f"No event log for session {session_id}.") from exc
            return []
        except OSError as exc:
            raise EventStoreError(f"Could not open log for session {session_id}.") from exc

        history: list[SessionEvent] = []
        known_ids: set[UUID] = set()
        try:
            with file:
                for line_number, line in enumerate(file, start=1):
                    try:
                        if not line.endswith(b"\n"):
                            raise ValueError("Unterminated final record (possible interrupted write).")
                        event = _decode(line)
                        check_event_position(event, session_id, line_number, known_ids)
                    except ValueError as exc:
                        raise EventStoreCorruptError(
                            session_id, line_number, len(history), str(exc),
                        ) from exc
                    history.append(event)
                    known_ids.add(event.event_id)
        except OSError as exc:
            raise EventStoreError(f"Could not read log for session {session_id}.") from exc
        return history

    def _session_lock(self, session_id: UUID, *, create: bool = True) -> AbstractContextManager[None]:
        lock_path = self.root / "locks" / f"{session_id}.lock"
        return file_lock(lock_path, f"Session {session_id}", create=create)


class SessionLogLease:
    """Operate on one log under a single held OS lock."""

    def __init__(self, store: JsonlEventStore, session_id: UUID) -> None:
        self._store = store
        self._path = store._session_path(session_id)
        self._session_id = session_id
        self._guard = RLock()
        self._closed = False
        self._lock = store._session_lock(session_id)
        self._lock.__enter__()

    @property
    def session_id(self) -> UUID:
        return self._session_id

    @property
    def closed(self) -> bool:
        with self._guard:
            return self._closed

    def __enter__(self) -> "SessionLogLease":
        with self._guard:
            self._check_session(self._session_id)
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    def close(self) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
            self._lock.__exit__(None, None, None)

    def iter_events(self, session_id: UUID, *, require_log: bool = False) -> Iterator[SessionEvent]:
        with self._guard:
            self._check_session(session_id)
            return iter(self._store._read(self._path, session_id, require_log=require_log))

    def append(self, event: SessionEvent) -> None:
        with self._guard:
            self._check_session(event.session_id)
            self._store._append_locked(self._path, event)

    def publish(self, session_id: UUID, events: Iterable[SessionEvent]) -> None:
        with self._guard:
            self._check_session(session_id)
        # Caller code in an iterable can close the lease before publication.
        snapshot = tuple(events)
        with self._guard:
            self._check_session(session_id)
            self._store._publish_locked(self._path, session_id, snapshot)

    def _check_session(self, session_id: UUID) -> None:
        if self._closed:
            raise EventStoreError("Session log lease is closed.")
        if session_id != self._session_id:
            raise EventStoreError("Session ID does not match the locked log.")


class JsonlEventSink:
    """Bind a store to the single session used by SessionEventWriter."""

    def __init__(self, store: JsonlEventStore | SessionLogLease, session_id: UUID) -> None:
        self._store = store
        self._session_id = session_id

    @property
    def history(self) -> ConversationHistory:
        return tuple(self._store.iter_events(self._session_id))

    def append(self, event: SessionEvent) -> None:
        if event.session_id != self._session_id:
            raise EventStoreError("Event session ID does not match the bound sink.")
        self._store.append(event)
