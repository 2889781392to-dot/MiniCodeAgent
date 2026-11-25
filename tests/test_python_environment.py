import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from minicodeagent import python_environment
from minicodeagent.protocol import ToolResult
from minicodeagent.tools import process
from minicodeagent.workspace import Workspace


def allow_probe(_argv, _cwd):
    return True


def candidate_venv(workspace):
    launch = (".venv", "Scripts", "python.exe") if os.name == "nt" else (".venv", "bin", "python")
    path = workspace.root.joinpath(*launch)
    path.parent.mkdir(parents=True)
    path.touch()
    return path


def result(call_id, *, status="ok", stdout=None, **details):
    if stdout is None:
        stdout = json.dumps({"python_version": "3.11.16", "pytest_discoverable": True})
    return ToolResult(
        call_id=call_id,
        status=status,
        output="probe result",
        details={"exit_code": 0, "stdout": stdout, **details},
    )


def test_explicit_external_launch_path_with_spaces_beats_workspace_venv(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    workspace = Workspace(project)
    candidate_venv(workspace)
    launch = tmp_path / "external Python" / "python.exe"
    launch.parent.mkdir()
    launch.touch()
    approved = []

    def confirm(argv, cwd):
        approved.append((argv, cwd))
        return True

    def fake_execute(argv, cwd, timeout_seconds, call_id, confirm):
        assert argv[0] == str(launch)
        assert argv[1:4] == ("-I", "-B", "-c")
        assert len(argv) == 5 and 0 < timeout_seconds <= 10
        assert confirm(argv, cwd)
        return result(call_id)

    monkeypatch.setattr(python_environment, "execute_command", fake_execute)
    observation = python_environment.discover_python_environment(
        workspace, explicit_python=launch, confirm=confirm,
    )

    assert len(approved) == 1
    assert approved[0][0][0] == str(launch) and approved[0][1] == workspace.root
    assert observation.workspace == workspace.root and observation.python == launch
    assert observation.source == "explicit" and observation.verification == "verified"
    assert observation.python_version == "3.11.16"
    assert observation.pytest_status == "discoverable"


def test_standard_venv_launch_path_is_preserved_even_if_it_resolves_elsewhere(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    launch = candidate_venv(workspace)
    actual_resolve = Path.resolve
    actual_python = Path(sys.executable)
    started = []

    def resolve(path, *args, **kwargs):
        if path == launch:
            return actual_python
        return actual_resolve(path, *args, **kwargs)

    def fake_execute(argv, _cwd, _timeout, call_id, _confirm):
        started.append(argv[0])
        return result(call_id, stdout='{"python_version":"3.11.16","pytest_discoverable":false}')

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(python_environment, "execute_command", fake_execute)
    observation = python_environment.discover_python_environment(workspace, confirm=allow_probe)

    assert started == [str(launch)]
    assert observation.python == launch and observation.source == "workspace_venv"
    assert observation.verification == "verified" and observation.pytest_status == "missing"


@pytest.mark.parametrize("venv_present", [False, True])
def test_automatic_selection_falls_back_to_runtime_only_after_missing_or_unlaunchable_venv(
    tmp_path, monkeypatch, venv_present,
):
    workspace = Workspace(tmp_path)
    venv = candidate_venv(workspace) if venv_present else None
    started = []

    def fake_execute(argv, cwd, _timeout, call_id, _confirm):
        started.append(argv[0])
        assert cwd == workspace.root
        if venv is not None and argv[0] == str(venv):
            return result(call_id, status="error", error_code="tool_error", execution_outcome="not_started")
        return result(call_id)

    monkeypatch.setattr(python_environment, "execute_command", fake_execute)
    observation = python_environment.discover_python_environment(workspace, confirm=allow_probe)

    assert started == ([str(venv)] if venv_present else []) + [sys.executable]
    assert observation.python == Path(sys.executable) and observation.source == "runtime_fallback"
    assert observation.verification == "verified"
    if venv_present:
        assert observation.diagnostics


@pytest.mark.parametrize("candidate_exists", [False, True])
def test_explicit_failure_reports_the_chosen_path_without_fallback(tmp_path, monkeypatch, candidate_exists):
    workspace = Workspace(tmp_path)
    candidate_venv(workspace)
    launch = tmp_path / "explicit.exe"
    if candidate_exists:
        launch.touch()
    started = []

    def fake_execute(argv, _cwd, _timeout, call_id, _confirm):
        started.append(argv[0])
        return result(call_id, status="error", error_code="tool_error", execution_outcome="not_started")

    monkeypatch.setattr(python_environment, "execute_command", fake_execute)
    observation = python_environment.discover_python_environment(
        workspace, explicit_python=launch, confirm=allow_probe,
    )

    assert started == ([str(launch)] if candidate_exists else [])
    assert observation.python == launch and observation.source == "explicit"
    assert observation.verification == "failed" and observation.pytest_status == "unverified"
    assert observation.python_version is None and observation.diagnostics


@pytest.mark.parametrize(
    ("status", "details", "diagnostic"),
    [
        ("timeout", {"error_code": "command_timeout"}, ("timeout", "timed out")),
        ("ok", {"stdout_truncated": True}, ("output", "truncat")),
        ("ok", {"stderr_truncated": True}, ("output", "truncat")),
        ("ok", {"stdout": "not a probe response"}, ("invalid", "malformed", "format")),
        ("ok", {"stdout": '{"python_version":"3.11.16","pytest_discoverable":"false"}'},
         ("invalid", "malformed", "format")),
    ],
)
def test_started_but_unusable_automatic_probe_does_not_become_verified(
    tmp_path, monkeypatch, status, details, diagnostic,
):
    workspace = Workspace(tmp_path)
    launch = candidate_venv(workspace)
    attempts = []

    def fake_execute(argv, _cwd, _timeout, call_id, _confirm):
        attempts.append(argv[0])
        return result(call_id, status=status, **details)

    monkeypatch.setattr(python_environment, "execute_command", fake_execute)
    observation = python_environment.discover_python_environment(workspace, confirm=allow_probe)

    assert attempts == [str(launch)]
    assert observation.python == launch and observation.source == "workspace_venv"
    assert observation.verification == "failed" and observation.pytest_status == "unverified"
    assert observation.python_version is None
    assert any(word in " ".join(observation.diagnostics).lower() for word in diagnostic)


def test_denied_automatic_probe_never_starts_a_process_or_tries_runtime(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    launch = candidate_venv(workspace)
    approvals = []

    def reject(argv, cwd):
        approvals.append((argv, cwd))
        return False

    def unexpected_popen(*_args, **_kwargs):
        pytest.fail("A denied environment probe started a subprocess.")

    monkeypatch.setattr(subprocess, "Popen", unexpected_popen)
    observation = python_environment.discover_python_environment(workspace, confirm=reject)

    assert len(approvals) == 1
    assert approvals[0][0][0] == str(launch) and approvals[0][1] == workspace.root
    assert observation.python == launch and observation.source == "workspace_venv"
    assert observation.verification == "unverified" and observation.pytest_status == "unverified"
    assert observation.diagnostics


def test_real_approved_probe_uses_an_external_python_and_reports_pytest_discoverable(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    launch = Path(sys.executable)
    allowed = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP", "USERPROFILE",
               "LOCALAPPDATA", "APPDATA", "SYSTEMDRIVE", "PROGRAMDATA")
    environ = {key: os.environ[key] for key in allowed if key in os.environ}
    environ.update({"OPENAI_API_KEY": "synthetic-key", "MINICODEAGENT_API_KEY": "synthetic-key"})
    monkeypatch.setattr(process, "os", SimpleNamespace(name=os.name, environ=environ))
    original_popen = subprocess.Popen
    trail = []

    def confirm(argv, cwd):
        trail.append(("approved", argv, cwd))
        return True

    def capture_popen(argv, **kwargs):
        trail.append(("started", argv, kwargs["cwd"]))
        assert kwargs["shell"] is False
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "MINICODEAGENT_API_KEY" not in kwargs["env"]
        return original_popen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", capture_popen)
    observation = python_environment.discover_python_environment(
        workspace, explicit_python=launch, confirm=confirm,
    )

    assert [step for step, *_ in trail] == ["approved", "started"]
    assert trail[0][1:] == trail[1][1:]
    assert trail[1][1][:4] == (str(launch), "-I", "-B", "-c")
    assert trail[1][2] == workspace.root
    assert observation.python == launch and observation.source == "explicit"
    assert observation.verification == "verified" and observation.python_version == platform.python_version()
    assert observation.pytest_status == "discoverable"
