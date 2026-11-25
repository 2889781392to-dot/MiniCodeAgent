"""Subprocess execution with bounded pipe capture and cooperative cleanup."""

import os
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from threading import Lock, Thread
from time import monotonic
from typing import BinaryIO
from uuid import UUID

from ..config import API_KEY_ENV
from ..environment import isolated_environment
from ..protocol import ToolResult

CommandConfirmation = Callable[[tuple[str, ...], Path], bool]
MAX_STREAM_BYTES = 8 * 1024
_CLEANUP_SECONDS = 1.0


class _Capture:
    def __init__(self) -> None:
        self.lock = Lock()
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0
        self.read_failed = False

    def drain(self, stream: BinaryIO) -> None:
        try:
            with stream:
                while chunk := stream.read(4096):
                    with self.lock:
                        self.total += len(chunk)
                        head_space = MAX_STREAM_BYTES // 2 - len(self.head)
                        self.head.extend(chunk[:head_space])
                        self.tail.extend(chunk[head_space:])
                        del self.tail[:max(0, len(self.tail) - MAX_STREAM_BYTES // 2)]
        except OSError:
            self.read_failed = True

    def render(self) -> tuple[str, bool]:
        with self.lock:
            head, tail, total = bytes(self.head), bytes(self.tail), self.total
        omitted = max(0, total - len(head) - len(tail))
        if omitted:
            marker = f"\n[output truncated: {omitted} bytes omitted]\n"
            return head.decode("utf-8", errors="replace") + marker + tail.decode("utf-8", errors="replace"), True
        return (head + tail).decode("utf-8", errors="replace"), False


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def execute_command(
    argv: tuple[str, ...],
    cwd: Path,
    timeout_seconds: float,
    call_id: UUID,
    confirm: CommandConfirmation,
) -> ToolResult:
    try:
        approved = confirm(argv, cwd)
    except KeyboardInterrupt:
        return ToolResult(
            call_id=call_id, status="interrupted", output="Interrupted before starting the command.",
            details={"execution_outcome": "not_started"},
        )
    except Exception:
        approved = False
    if approved is not True:
        return ToolResult(
            call_id=call_id, status="error", output="Command confirmation was denied or unavailable.",
            details={"error_code": "command_denied", "execution_outcome": "not_started"},
        )

    # Approval covers the exact resolved directory shown to the user.
    try:
        cwd_unchanged = cwd.resolve(strict=True) == cwd and cwd.is_dir()
    except (OSError, RuntimeError):
        cwd_unchanged = False
    if not cwd_unchanged:
        return ToolResult(
            call_id=call_id, status="error", output="The approved command directory changed before launch.",
            details={"error_code": "tool_error", "execution_outcome": "not_started"},
        )

    # Commands should not inherit the credential used by the API client.
    env = isolated_environment(os.environ, omit_key=API_KEY_ENV)
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
            start_new_session=os.name == "posix",
        )
    except OSError:
        return ToolResult(
            call_id=call_id, status="error", output="Could not start the command. Check the executable and cwd.",
            details={"error_code": "tool_error", "execution_outcome": "not_started"},
        )

    deadline = monotonic() + timeout_seconds
    stdout, stderr = _Capture(), _Capture()
    readers: list[Thread] = []
    result_status = "ok"
    try:
        for capture, stream in ((stdout, process.stdout), (stderr, process.stderr)):
            assert stream is not None
            worker = Thread(target=capture.drain, args=(stream,), daemon=True)
            worker.start()
            readers.append(worker)
        process.wait(timeout=max(0, deadline - monotonic()))
        for reader in readers:
            reader.join(timeout=max(0, deadline - monotonic()))
        if any(reader.is_alive() for reader in readers):
            raise subprocess.TimeoutExpired(argv, timeout_seconds)
    except subprocess.TimeoutExpired:
        result_status = "timeout"
    except KeyboardInterrupt:
        result_status = "interrupted"
    except Exception:
        _stop_process(process)
        raise

    if result_status != "ok":
        _stop_process(process)
        cleanup_deadline = monotonic() + _CLEANUP_SECONDS
        for reader in readers:
            reader.join(timeout=max(0, cleanup_deadline - monotonic()))

    capture_incomplete = (
        len(readers) != 2 or any(reader.is_alive() for reader in readers)
        or stdout.read_failed or stderr.read_failed
    )
    out, stdout_truncated = stdout.render()
    err, stderr_truncated = stderr.render()
    details = {
        "exit_code": process.returncode,
        "stdout": out,
        "stderr": err,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "capture_incomplete": capture_incomplete,
    }
    if result_status == "timeout":
        summary = f"Command timed out after {timeout_seconds:g} seconds."
        details["error_code"] = "command_timeout"
        details["execution_outcome"] = "unknown"
    elif result_status == "interrupted":
        summary = "Command interrupted."
        details["execution_outcome"] = "unknown"
    else:
        summary = f"Exit code: {process.returncode}"
        if process.returncode != 0 or details["capture_incomplete"]:
            result_status = "error"
            details["error_code"] = "command_failed"
    return ToolResult(
        call_id=call_id,
        status=result_status,
        output=f"{summary}\nstdout:\n{out}\nstderr:\n{err}",
        details=details,
    )
