"""Observe the Python interpreter used for this run's actual Workspace."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal
from uuid import uuid4

from .tools.process import CommandConfirmation, execute_command
from .workspace import Workspace

_PROBE_TIMEOUT_SECONDS = 5.0
_PROBE = (
    "import importlib.util,json,sys; "
    "print(json.dumps({'python_version': '.'.join(map(str, sys.version_info[:3])), "
    "'pytest_discoverable': importlib.util.find_spec('pytest') is not None}))"
)
PythonSource = Literal["explicit", "workspace_venv", "runtime_fallback"]


@dataclass(frozen=True, slots=True)
class PythonEnvironmentObservation:
    workspace: Path
    python: Path | None
    source: PythonSource
    verification: Literal["verified", "failed", "unverified"] = "unverified"
    python_version: str | None = None
    pytest_status: Literal["discoverable", "missing", "unverified"] = "unverified"
    diagnostics: tuple[str, ...] = ()

    def prompt(self) -> str:
        executable = str(self.python) if self.python is not None else None
        observation = json.dumps({
            "workspace": str(self.workspace),
            "python_executable": executable,
            "source": self.source,
            "verification": self.verification,
            "python_version": self.python_version,
            "pytest_status": self.pytest_status,
            "diagnostics": self.diagnostics,
        }, ensure_ascii=False)
        notes = [
            "runtime_fallback is the MiniCodeAgent runtime interpreter.",
            "pytest_status describes module discoverability in the isolated probe.",
        ]
        if executable is not None and self.verification == "verified" and self.pytest_status == "discoverable":
            argv = json.dumps([executable, "-m", "pytest"], ensure_ascii=False)
            notes.append(f"Suggested run_command argv {argv}. Use the usual command confirmation.")
        return "\n".join(("[Current run environment observation]", observation, *notes,
                          "[End current run environment observation]"))


def discover_python_environment(
    workspace: Workspace, *, explicit_python: Path | None = None, confirm: CommandConfirmation,
) -> PythonEnvironmentObservation:
    if explicit_python is not None:
        if not explicit_python.is_absolute():
            raise ValueError("--python requires an absolute executable path.")
        observation, _ = _observe_candidate(workspace, explicit_python, "explicit", confirm)
        return observation

    launcher = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    venv = workspace.root.joinpath(".venv", *launcher)
    observation, launch_failed = _observe_candidate(workspace, venv, "workspace_venv", confirm)
    if not launch_failed:
        return observation

    runtime = Path(sys.executable).absolute() if sys.executable else None
    fallback, _ = _observe_candidate(workspace, runtime, "runtime_fallback", confirm)
    return replace(fallback, diagnostics=(*observation.diagnostics, *fallback.diagnostics))


def _observe_candidate(
    workspace: Workspace, python: Path | None, source: PythonSource, confirm: CommandConfirmation,
) -> tuple[PythonEnvironmentObservation, bool]:
    """Return the observation and whether this candidate could not start."""
    observation = PythonEnvironmentObservation(workspace.root, python, source)

    def failed(message: str, *, launch_failed: bool = False) -> tuple[PythonEnvironmentObservation, bool]:
        return replace(
            observation, verification="failed", diagnostics=(f"{source} ({python}): {message}",),
        ), launch_failed

    try:
        if python is None or not python.is_file():
            return failed("Interpreter does not exist or is not a file.", launch_failed=True)
    except OSError:
        return failed("Could not access the interpreter path.", launch_failed=True)

    # Keep the launch path, including a virtual environment's symlink spelling.
    argv = (str(python), "-I", "-B", "-c", _PROBE)
    try:
        result = execute_command(argv, workspace.root, _PROBE_TIMEOUT_SECONDS, uuid4(), confirm)
    except (OSError, RuntimeError):
        return failed("Interpreter probe could not complete in the command runner.")
    if result.status == "interrupted":
        raise KeyboardInterrupt
    details = result.details
    if details.get("error_code") == "command_denied":
        return replace(observation, diagnostics=("Probe confirmation was denied or unavailable.",)), False
    if result.status == "timeout":
        return failed(f"Interpreter probe timed out after {_PROBE_TIMEOUT_SECONDS:g} seconds.")
    if details.get("error_code") == "tool_error" and details.get("execution_outcome") == "not_started":
        return failed("Could not start the interpreter probe. Check the executable and working directory.",
                      launch_failed=True)
    if details.get("stdout_truncated") or details.get("stderr_truncated"):
        return failed("Interpreter probe output exceeded the capture limit.")
    if details.get("capture_incomplete"):
        return failed("Interpreter probe output capture was incomplete.")
    if result.status != "ok" or details.get("exit_code") != 0:
        return failed("Interpreter probe exited unsuccessfully.")

    parsed = _parse_probe(details.get("stdout"))
    if parsed is None:
        return failed("Interpreter probe returned invalid output.")
    version, pytest_found = parsed
    return replace(observation, verification="verified", python_version=version,
                   pytest_status="discoverable" if pytest_found else "missing"), False


def _parse_probe(output: object) -> tuple[str, bool] | None:
    if not isinstance(output, str):
        return None
    try:
        payload = json.loads(output)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.keys() != {"python_version", "pytest_discoverable"}:
        return None
    version = payload["python_version"]
    pytest_found = payload["pytest_discoverable"]
    if (
        not isinstance(version, str) or len(version) > 32
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None
        or type(pytest_found) is not bool
    ):
        return None
    return version, pytest_found
