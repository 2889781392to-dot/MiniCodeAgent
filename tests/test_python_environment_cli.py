"""Exercise the environment snapshot at the real CLI/model/tool boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

from minicodeagent import cli, python_environment
from minicodeagent.context.events import (
    AssistantMessagePayload,
    ModelRequestStartedPayload,
    UserMessagePayload,
)
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.protocol import ModelError, ModelReply, ToolCall, ToolResult
from minicodeagent.session.service import SessionService


OBSERVATION_MARKER = "[Current run environment observation]"


def capture_model(monkeypatch, respond=None):
    requests = []

    class FakeModel:
        def __init__(self, _settings):
            pass

        def complete(self, request):
            requests.append(request)
            if respond is not None:
                return respond(request, len(requests))
            return ModelReply(content="done", finish_reason="stop")

        def close(self):
            pass

    monkeypatch.setenv("MINICODEAGENT_API_KEY", "synthetic-environment-test-key")
    monkeypatch.setenv("MINICODEAGENT_MODEL", "fake-environment-model")
    monkeypatch.setattr(cli, "LLMClient", FakeModel)
    return requests


def observation(request):
    system, = [message for message in request.messages if message.role == "system"]
    assert system.content.count(OBSERVATION_MARKER) == 1
    assert all(
        OBSERVATION_MARKER not in item.content for item in request.messages if item.role != "system"
    )
    suffix = system.content.split(OBSERVATION_MARKER, 1)[1].lstrip()
    recorded, _ = json.JSONDecoder().raw_decode(suffix)
    return recorded


def resume(data_root, session_id, *args):
    return cli.main(["resume", str(session_id), "--data-root", str(data_root), *args])


def venv_candidate(workspace):
    suffix = Path(".venv/Scripts/python.exe") if os.name == "nt" else Path(".venv/bin/python")
    return workspace / suffix


def probe_success(call_id, version, *, has_pytest):
    return ToolResult(
        call_id=call_id,
        status="ok",
        output="",
        details={
            "exit_code": 0,
            "stdout": json.dumps({"python_version": version, "pytest_discoverable": has_pytest}),
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "capture_incomplete": False,
        },
    )


def test_cli_exposes_explicit_external_path_and_project_memory(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "work"
    workspace.mkdir()
    source = workspace / "AGENTS.md"
    source.write_text("Use the shared project convention.", encoding="utf-8")
    local = venv_candidate(workspace)
    local.parent.mkdir(parents=True)
    local.touch()
    executable_name = "python executable.exe" if os.name == "nt" else "python executable"
    selected = tmp_path / "external environment" / executable_name
    selected.parent.mkdir()
    selected.touch()
    requests = capture_model(monkeypatch)
    confirmations = []

    def approve(argv, cwd):
        confirmations.append((argv, cwd))
        return True

    def execute(argv, cwd, timeout_seconds, call_id, confirm):
        assert confirm(argv, cwd)
        assert timeout_seconds > 0
        return probe_success(call_id, "3.11.16", has_pytest=False)

    monkeypatch.setattr(cli, "confirm_command", approve)
    monkeypatch.setattr(python_environment, "execute_command", execute)
    argv = ["--workspace", str(workspace), "--python", str(selected), "--message", "task"]
    assert cli.main(argv) == 0

    assert len(confirmations) == len(requests) == 1
    argv, cwd = confirmations[0]
    assert argv[:4] == (str(selected), "-I", "-B", "-c")
    assert cwd == workspace.resolve()
    received = observation(requests[0])
    assert received["workspace"] == str(workspace.resolve())
    assert received["python_executable"] == str(selected)
    assert received["source"] == "explicit"
    assert received["verification"] == "verified"
    assert received["python_version"] == "3.11.16"
    assert received["pytest_status"] == "missing"
    system = requests[0].messages[0].content
    assert system.count("[Project background from Workspace AGENTS.md]") == 1
    assert system.count("Use the shared project convention.") == 1
    assert "missing" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == "Use the shared project convention."


def test_resume_override_and_forks_follow_effective_workspace(tmp_path, monkeypatch, capsys):
    original = tmp_path / "original"
    other = tmp_path / "temporary or branch"
    original.mkdir()
    other.mkdir()
    for root, marker in ((original, "Original project memory."), (other, "Other project memory.")):
        (root / "AGENTS.md").write_text(marker, encoding="utf-8")
    agent_files = {path: path.read_bytes() for path in tmp_path.rglob("AGENTS.md")}

    data_root = tmp_path / "sessions"
    store = JsonlEventStore(data_root)
    service = SessionService(store)
    requests = capture_model(monkeypatch)
    interpreter = str(Path(sys.executable).absolute())
    approvals = []
    spawns = []
    original_spawn = subprocess.Popen

    def approve(prompt):
        assert prompt == "Run this command once? [y/N] "
        approvals.append(prompt)
        return "y"

    def spawn(argv, **kwargs):
        spawns.append((list(argv), kwargs["cwd"]))
        assert kwargs["shell"] is False
        return original_spawn(argv, **kwargs)

    monkeypatch.setattr("builtins.input", approve)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    assert cli.main(["new", "--data-root", str(data_root), "--workspace", str(original)]) == 0
    parent_id = UUID(capsys.readouterr().out.strip())
    assert resume(data_root, parent_id, "--message", "original task") == 0
    output = capsys.readouterr().err
    assert resume(data_root, parent_id, "--workspace", str(other), "--message", "temporary") == 0
    override_output = capsys.readouterr().err
    assert "Project Memory is paused" in override_output
    output += override_output
    parent_path = data_root / "sessions" / f"{parent_id}.jsonl"
    parent_bytes = parent_path.read_bytes()
    parent_history = tuple(store.iter_events(parent_id))

    fork = ["fork", str(parent_id), "--data-root", str(data_root)]
    assert cli.main(fork) == 0
    default_fork = UUID(capsys.readouterr().out.strip())
    assert cli.main([*fork, "--workspace", str(other)]) == 0
    override_fork = UUID(capsys.readouterr().out.strip())
    assert len({parent_id, default_fork, override_fork}) == 3
    assert len(approvals) == len(spawns) == 2
    assert parent_path.read_bytes() == parent_bytes

    histories = [parent_history]
    for child_id, root in ((default_fork, original), (override_fork, other)):
        child_history = tuple(store.iter_events(child_id))
        assert all(event.session_id == child_id for event in child_history)
        with service.open(child_id) as child:
            assert child.workspace.root == root.resolve()
            assert child.state.metadata.parent_session_id == parent_id
            assert child.state.metadata.parent_through_seq == parent_history[-1].seq
        assert [event.payload for event in child_history[1:]] == [event.payload for event in parent_history[1:]]
        histories.append(child_history)
    ids = [event.event_id for history in histories for event in history]
    assert len(ids) == len(set(ids))

    assert resume(data_root, default_fork, "--message", "shared") == 0
    output += capsys.readouterr().err
    assert resume(data_root, override_fork, "--python", interpreter, "--message", "branch") == 0
    output += capsys.readouterr().err

    roots = (original, other, original, other)
    assert len(requests) == len(roots)
    for req, root, marker, source in zip(
        requests, roots, ("Original", None, "Original", "Other"),
        ("runtime_fallback", "runtime_fallback", "runtime_fallback", "explicit"), strict=True,
    ):
        env = observation(req)
        assert env["workspace"] == str(root.resolve())
        assert env["python_executable"] == interpreter
        assert env["source"] == source
        assert env["verification"] == "verified"
        if marker is None:
            assert "[Project background from Workspace AGENTS.md]" not in req.messages[0].content
        else:
            assert req.messages[0].content.count(f"{marker} project memory.") == 1
        assert ("Original project memory." in req.messages[0].content) == (marker == "Original")
        assert ("Other project memory." in req.messages[0].content) == (marker == "Other")

    shown_cwd = [json.loads(line.removeprefix("Command cwd: ")) for line in output.splitlines()
                 if line.startswith("Command cwd: ")]
    shown_argv = [json.loads(line.removeprefix("Command argv: ")) for line in output.splitlines()
                  if line.startswith("Command argv: ")]
    assert len(approvals) == len(spawns) == len(roots)
    assert spawns == list(zip(shown_argv, map(Path, shown_cwd), strict=True))
    assert [cwd for _, cwd in spawns] == [root.resolve() for root in roots]
    assert all(argv[:4] == [interpreter, "-I", "-B", "-c"] for argv, _ in spawns)

    assert parent_path.read_bytes() == parent_bytes
    assert {path: path.read_bytes() for path in tmp_path.rglob("AGENTS.md")} == agent_files
    with service.open(parent_id) as handle:
        assert handle.state.metadata.workspace == str(original.resolve())


def test_denied_noninteractive_probe_keeps_task_running_without_spawn(tmp_path, monkeypatch, capsys):
    requests = capture_model(monkeypatch)

    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("a noninteractive probe launched a process")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(subprocess, "Popen", unexpected_spawn)
    assert cli.main(["--workspace", str(tmp_path), "--message", "task"]) == 0
    received = observation(requests[0])
    assert received["source"] == "runtime_fallback"
    assert received["python_executable"] == str(Path(sys.executable).absolute())
    assert received["verification"] == received["pytest_status"] == "unverified"
    assert any("denied or unavailable" in reason for reason in received["diagnostics"])
    assert "Interactive confirmation is unavailable" in capsys.readouterr().err


def test_log_commands_replay_and_saved_reply_do_not_probe(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "project"
    workspace.mkdir()
    agents = workspace / "AGENTS.md"
    agents.write_bytes(b"Background for active runs only.")
    data_root = tmp_path / "sessions"
    store = JsonlEventStore(data_root)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a log-only operation prepared a model or probe")

    monkeypatch.setattr(python_environment, "execute_command", forbidden)
    monkeypatch.setattr(cli, "confirm_command", forbidden)
    monkeypatch.setattr(cli, "LLMClient", forbidden)
    assert cli.main(["new", "--data-root", str(data_root), "--workspace", str(workspace)]) == 0
    parent_id = UUID(capsys.readouterr().out.strip())
    service = SessionService(store)

    with service.open(parent_id) as handle:
        turn_id = uuid4()
        request_id = uuid4()
        handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="old request"))
        handle.writer.append("model_request_started", ModelRequestStartedPayload(
            turn_id=turn_id, request_id=request_id, attempt=1, trigger="initial",
        ))
        handle.writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn_id, request_id=request_id, content="stored answer", finish_reason="stop",
        ))
    log_path = data_root / "sessions" / f"{parent_id}.jsonl"
    before = log_path.read_bytes()

    assert cli.main(["list", "--data-root", str(data_root)]) == 0
    assert cli.main(["fork", str(parent_id), "--data-root", str(data_root)]) == 0
    replay = SessionReplay(store).replay(parent_id, "fixed")
    assert replay.state.current_turn.finished is None
    assert log_path.read_bytes() == before

    capsys.readouterr()
    assert resume(data_root, parent_id, "--continue") == 0
    assert "stored answer" in capsys.readouterr().out
    assert log_path.read_bytes().startswith(before)
    assert [event.type for event in store.iter_events(parent_id)] == [
        "session_created", "user_message", "model_request_started", "assistant_message", "turn_finished",
    ]
    assert agents.read_bytes() == b"Background for active runs only."


def test_repl_real_pytest_uses_selected_argv_and_only_one_probe(tmp_path, monkeypatch):
    workspace = tmp_path / "sample project"
    workspace.mkdir()
    source = workspace / "AGENTS.md"
    source.write_text("Snapshot from the first turn.", encoding="utf-8")
    python = Path(sys.executable).absolute()
    project_test = workspace / "test_runtime.py"
    project_test.write_text(
        "import sys\nfrom pathlib import Path\n\n"
        "def test_selected_interpreter():\n"
        f"    assert Path(sys.executable) == Path({str(python)!r})\n",
        encoding="utf-8",
    )
    pytest_argv = [
        str(python), "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--basetemp", str(tmp_path / "nested pytest artifacts"),
    ]
    approvals = []
    spawns = []
    original_spawn = subprocess.Popen

    def approve(argv, cwd):
        approvals.append((argv, cwd))
        return True

    def spawn(argv, **options):
        spawns.append((tuple(argv), options["cwd"]))
        assert options["shell"] is False
        assert not any(name.upper().startswith("OPENAI_") for name in options["env"])
        assert "MINICODEAGENT_API_KEY" not in options["env"]
        return original_spawn(argv, **options)

    def respond(request, count):
        info = observation(request)
        assert info["python_executable"] == str(python)
        assert info["verification"] == "verified" and info["pytest_status"] == "discoverable"
        assert request.messages[0].content.count("[Project background from Workspace AGENTS.md]") == 1
        if count <= 3:
            assert "Snapshot from the first turn." in request.messages[0].content
        if count == 1:
            source.write_text("Refreshed project rules.", encoding="utf-8")
            raise ModelError("timeout", "retry")
        if count == 2:
            argv = [info["python_executable"], *pytest_argv[1:]]
            return ModelReply(
                finish_reason="tool_calls",
                tool_calls=(ToolCall(
                    provider_call_id="test-argv", name="run_command", arguments_raw=json.dumps({"argv": argv}),
                ),),
            )
        if count == 3:
            assert request.messages[-1].role == "tool"
            assert "1 passed" in request.messages[-1].content
            assert "Exit code: 0" in request.messages[-1].content
        if count == 4:
            assert "Refreshed project rules." in request.messages[0].content
        return ModelReply(content="done", finish_reason="stop")

    requests = capture_model(monkeypatch, respond)
    tasks = iter(("test the project", "check the new project rules", "/exit"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(tasks))
    monkeypatch.setattr(cli, "confirm_command", approve)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    assert cli.main(["--workspace", str(workspace), "--python", str(python)]) == 0
    assert len(requests) == 4 and requests[0] == requests[1]
    assert len(approvals) == len(spawns) == 2
    assert spawns == approvals
    assert approvals[0][0][:4] == (str(python), "-I", "-B", "-c")
    assert approvals[1] == (tuple(pytest_argv), workspace.resolve())
