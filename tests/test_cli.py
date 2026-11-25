import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from minicodeagent import __main__ as entrypoint
from minicodeagent import cli
from minicodeagent.cli import confirm_command, main
from minicodeagent.config import Settings
from minicodeagent.context.events import ToolResultPayload
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.protocol import ModelReply, ToolCall


def test_cli_help_and_missing_configuration(monkeypatch, capsys):
    monkeypatch.delenv("MINICODEAGENT_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--message" in capsys.readouterr().out
    assert main(["--message", "task"]) == 2
    assert capsys.readouterr().err.strip() == "Set MINICODEAGENT_API_KEY before running MiniCodeAgent."


def test_cli_assembles_single_task(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MINICODEAGENT_API_KEY", "secret-key-example")
    monkeypatch.setenv("MINICODEAGENT_MODEL", "deepseek4.1f")
    monkeypatch.delenv("MINICODEAGENT_DEEPSEEK_THINKING", raising=False)
    seen = []

    class FakeClient:
        def __init__(self, settings):
            seen.append(settings.model_name)

        def complete(self, request):
            assert request.messages[-1].content == "Say hello"
            assert len(request.tools) == 6
            return ModelReply(content="Hello", finish_reason="stop")

        def close(self):
            seen.append("closed")

    monkeypatch.setattr("minicodeagent.cli.LLMClient", FakeClient)
    assert main(["--workspace", str(tmp_path), "--message", "Say hello"]) == 0
    output = capsys.readouterr()
    assert "Hello" in output.out and "[completed] model_stop" in output.out
    assert "secret-key-example" not in output.out + output.err
    assert seen == ["deepseek4.1f", "closed"]


def test_cli_entrypoint_excludes_ambient_sdk_settings_without_reading_values(monkeypatch):
    class Ambient(dict):
        def __getitem__(self, name):
            assert not entrypoint.is_openai_variable(name)
            return super().__getitem__(name)

    ambient = Ambient({
        "OPENAI_API_KEY": "synthetic-sdk-key",
        "openai_custom_headers": "synthetic-header",
        "MINICODEAGENT_API_KEY": "synthetic-project-key",
        "PATH": "synthetic-path",
    })
    monkeypatch.setattr(entrypoint, "os", SimpleNamespace(environ=ambient))
    def launch(argv, *, env):
        assert argv == [sys.executable, "-m", "minicodeagent", "--help"]
        assert env == {"MINICODEAGENT_API_KEY": "synthetic-project-key", "PATH": "synthetic-path"}
        return 23

    monkeypatch.setattr(entrypoint, "subprocess", SimpleNamespace(call=launch))
    assert entrypoint.main(["--help"]) == 23
    assert list(ambient) == ["OPENAI_API_KEY", "openai_custom_headers", "MINICODEAGENT_API_KEY", "PATH"]


def test_confirmation_displays_exact_argv_before_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    def answer(prompt):
        assert prompt == "Run this command once? [y/N] "
        output = capsys.readouterr().err
        assert '["python", "argument with spaces"]' in output
        assert "Command cwd:" in output
        return "yes"
    monkeypatch.setattr("builtins.input", answer)
    assert confirm_command(("python", "argument with spaces"), tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert confirm_command(("python", "argument with spaces"), tmp_path) is False


def test_session_new_and_list_use_no_model_configuration(tmp_path, monkeypatch, capsys):
    def unexpected_model(*_args, **_kwargs):
        pytest.fail("Session commands must not initialize a model.")

    monkeypatch.setattr("minicodeagent.cli.LLMClient", unexpected_model)
    monkeypatch.setattr("minicodeagent.cli.Settings.from_env", unexpected_model)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_root = tmp_path / "data"
    options = ["--data-root", str(data_root)]
    assert main(["new", *options, "--workspace", str(workspace), "--title", "example"]) == 0
    session_id = UUID(capsys.readouterr().out.strip())

    assert main(["list", *options]) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output) == 1
    assert output[0]["id"] == str(session_id)
    assert output[0]["workspace"] == str(workspace.resolve())
    assert output[0]["title"] == "example"
    assert output[0]["status"] == "valid"


def test_resume_and_fork_commands_use_persisted_history(tmp_path, monkeypatch, capsys):
    from minicodeagent.context.store import JsonlEventStore
    from minicodeagent.session.service import SessionService

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_root = tmp_path / "data"
    options = ["--data-root", str(data_root)]
    service = SessionService(JsonlEventStore(data_root))
    with service.create(workspace, "parent") as parent:
        source_id = parent.session_id
    monkeypatch.setenv("MINICODEAGENT_API_KEY", "synthetic-key")
    monkeypatch.setenv("MINICODEAGENT_MODEL", "fake-model")
    messages = []

    class FakeClient:
        def __init__(self, _settings):
            pass

        def complete(self, request):
            messages.extend(item.content for item in request.messages)
            return ModelReply(content="Persisted reply", finish_reason="stop")

        def close(self):
            pass

    monkeypatch.setattr("minicodeagent.cli.LLMClient", FakeClient)
    assert main(["resume", str(source_id), *options, "--message", "Persist this"]) == 0
    assert "[completed] model_stop" in capsys.readouterr().out
    assert messages[-1] == "Persist this"
    with service.open(source_id) as parent:
        assert [event.type for event in parent.state.history] == [
            "session_created", "user_message", "model_request_started", "assistant_message", "turn_finished",
        ]
    path = data_root / "sessions" / f"{source_id}.jsonl"
    original = path.read_bytes()
    assert main(["fork", str(source_id), *options, "--title", "child"]) == 0
    child_id = UUID(capsys.readouterr().out.strip())
    assert child_id != source_id and path.read_bytes() == original
    with service.open(child_id) as child:
        assert child.state.metadata.parent_session_id == source_id
        assert child.state.current_turn.finished.reason == "model_stop"


def test_resume_refuses_pending_without_interactive_confirmation(tmp_path, monkeypatch, capsys):
    from uuid import uuid4
    from minicodeagent.context.events import AssistantMessagePayload, UserMessagePayload
    from minicodeagent.context.store import JsonlEventStore
    from minicodeagent.protocol import ToolCall
    from minicodeagent.session.service import SessionService

    data_root = tmp_path / "data"
    service = SessionService(JsonlEventStore(data_root))
    with service.create(tmp_path) as handle:
        session_id = handle.session_id
        turn_id = uuid4()
        handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="run"))
        handle.writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn_id, request_id=uuid4(), content="", finish_reason="tool_calls",
            tool_calls=(ToolCall(provider_call_id="wire", name="run_command", arguments_raw="{}"),),
        ))
    path = data_root / "sessions" / f"{session_id}.jsonl"
    original = path.read_bytes()

    def unexpected_client(*_args):
        pytest.fail("A rejected recovery must not assemble the model client.")

    monkeypatch.setattr("minicodeagent.cli.LLMClient", unexpected_client)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert main(["resume", str(session_id), "--data-root", str(data_root), "--message", "next"]) == 2
    assert "unknown" in capsys.readouterr().err.lower()
    assert path.read_bytes() == original


def test_resume_continue_reuses_the_registered_turn(tmp_path, monkeypatch, capsys):
    from uuid import uuid4
    from minicodeagent.context.events import ModelRequestStartedPayload, UserMessagePayload
    from minicodeagent.context.store import JsonlEventStore
    from minicodeagent.session.service import SessionService

    data_root = tmp_path / "data"
    service = SessionService(JsonlEventStore(data_root))
    turn_id = uuid4()
    with service.create(tmp_path) as handle:
        session_id = handle.session_id
        handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="original"))
        handle.writer.append("model_request_started", ModelRequestStartedPayload(
            turn_id=turn_id, request_id=uuid4(), attempt=1, trigger="initial",
        ))

    class FakeClient:
        def __init__(self, _settings):
            pass

        def complete(self, request):
            assert [item.content for item in request.messages if item.role == "user"] == ["original"]
            return ModelReply(content="Recovered", finish_reason="stop")

        def close(self):
            pass

    monkeypatch.setenv("MINICODEAGENT_API_KEY", "synthetic-key")
    monkeypatch.setenv("MINICODEAGENT_MODEL", "fake-model")
    monkeypatch.setattr("minicodeagent.cli.LLMClient", FakeClient)
    assert main(["resume", str(session_id), "--data-root", str(data_root), "--continue"]) == 0
    assert "Recovered" in capsys.readouterr().out
    with service.open(session_id) as handle:
        events = handle.state.history
        attempts = [event.payload for event in events if event.type == "model_request_started"]
        assert [attempt.attempt for attempt in attempts] == [1, 2]
        assert attempts[1].trigger == "resume" and attempts[1].turn_id == turn_id
        assert handle.state.current_turn.finished.turn_id == turn_id


def test_resume_recovers_empty_stop_as_a_saved_failure_without_model(tmp_path, monkeypatch, capsys):
    from uuid import uuid4
    from minicodeagent.context.events import AssistantMessagePayload, UserMessagePayload
    from minicodeagent.context.store import JsonlEventStore
    from minicodeagent.session.service import SessionService

    data_root = tmp_path / "data"
    service = SessionService(JsonlEventStore(data_root))
    with service.create(tmp_path) as handle:
        session_id = handle.session_id
        turn_id = uuid4()
        handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="original"))
        reply = handle.writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn_id, request_id=uuid4(), content="", finish_reason="stop",
        ))

    def unexpected_model(*_args, **_kwargs):
        pytest.fail("A saved terminal reply must not require a model.")

    monkeypatch.setattr("minicodeagent.cli.Settings.from_env", unexpected_model)
    monkeypatch.setattr("minicodeagent.cli.LLMClient", unexpected_model)
    assert main(["resume", str(session_id), "--data-root", str(data_root), "--continue"]) == 1
    assert "[failed] unexpected_finish_reason" in capsys.readouterr().out
    with service.open(session_id) as handle:
        assert handle.state.current_turn.finished.assistant_seq == reply.seq


@pytest.fixture
def persistent_cli_session(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = JsonlEventStore(tmp_path / "data")
    options = ["--data-root", str(store.root)]
    assert main(["new", *options, "--workspace", str(workspace)]) == 0
    session_id = UUID(capsys.readouterr().out.strip())
    settings = Settings.from_env({
        "MINICODEAGENT_API_KEY": "synthetic-cli-key",
        "MINICODEAGENT_MODEL": "fake-cli-model",
    })
    monkeypatch.setattr(cli.Settings, "from_env", lambda _env: settings)
    return workspace, store, session_id, ["resume", str(session_id), *options]


@pytest.mark.parametrize(
    ("rules", "diagnostic"),
    [(None, None), (b"x" * (16 * 1024 + 1), "limit"), (b"\xff", "utf-8")],
    ids=["missing-agents", "oversized-agents", "invalid-utf8-agents"],
)
def test_persistent_cli_project_memory_errors_stop_before_observation_and_request(
    persistent_cli_session, monkeypatch, capsys, rules, diagnostic,
):
    workspace, store, session_id, resume = persistent_cli_session
    source = workspace / "AGENTS.md"
    if rules is not None:
        source.write_bytes(rules)
    path = store.root / "sessions" / f"{session_id}.jsonl"
    before = path.read_bytes()
    missing_python = workspace / "missing Python"
    probe = Mock(wraps=cli.discover_python_environment)
    client = SimpleNamespace(
        complete=Mock(return_value=ModelReply(content="Task complete", finish_reason="stop")),
        close=Mock(),
    )
    monkeypatch.setattr(cli, "discover_python_environment", probe)
    monkeypatch.setattr(cli, "LLMClient", lambda _settings: client)
    result = main([*resume, "--python", str(missing_python), "--message", "Inspect the project"])
    output = capsys.readouterr()

    if diagnostic is not None:
        assert result == 1
        assert "[failed] project_memory_error" in output.out
        assert diagnostic in output.err.lower()
        assert source.read_bytes() == rules
        assert path.read_bytes() == before
        probe.assert_not_called()
        client.complete.assert_not_called()
        return

    assert result == 0
    assert not source.exists()
    probe.assert_called_once()
    request = client.complete.call_args.args[0]
    system = request.messages[0].content
    assert "[Project background from Workspace AGENTS.md]" not in system
    assert "[Current run environment observation]" in system
    assert '"verification": "failed"' in system
    assert "Interpreter does not exist" in system
    assert path.read_bytes().startswith(before)
    replay = SessionReplay(store).replay(session_id, "Replay history.")
    assert replay.state.current_turn.finished.reason == "model_stop"


def test_persistent_cli_records_real_timeout_before_model_followup(
    persistent_cli_session, monkeypatch, capsys,
):
    workspace, store, session_id, resume = persistent_cli_session
    argv = [
        sys.executable, "-B", "-u", "-c",
        "import time; print('child reached the timeout path'); time.sleep(20)",
    ]
    call = ToolCall(
        provider_call_id="timeout-command",
        name="run_command",
        arguments_raw=json.dumps({"argv": argv, "timeout_seconds": 1.5}),
    )
    path = store.root / "sessions" / f"{session_id}.jsonl"
    approvals = []
    requests = []

    def approve(prompt):
        assert prompt == "Run this command once? [y/N] "
        lines = capsys.readouterr().err.splitlines()
        cwd, = [json.loads(line.removeprefix("Command cwd: ")) for line in lines if line.startswith("Command cwd: ")]
        approved, = [
            json.loads(line.removeprefix("Command argv: ")) for line in lines if line.startswith("Command argv: ")
        ]
        approvals.append((cwd, approved))
        return "yes"

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return ModelReply(tool_calls=(call,), finish_reason="tool_calls")
        if len(requests) != 2:
            pytest.fail("A timed-out command must not restart.")
        # The CLI still holds the write lease, so observe what is already durable directly.
        events = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert [event["type"] for event in events[-2:]] == ["tool_result", "model_request_started"]
        result = events[-2]["payload"]
        assert result["status"] == "timeout" and result["details"]["execution_outcome"] == "unknown"
        assert "child reached the timeout path" in result["details"]["stdout"]
        assert result["details"]["error_code"] == "command_timeout"
        expected = request.messages[-1]
        assert expected.role == "tool" and expected.call_id == call.call_id
        assert result["call_id"] == str(expected.call_id)
        assert result["output"] == expected.content
        return ModelReply(content="The test command timed out.", finish_reason="stop")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", approve)
    monkeypatch.setattr(cli, "LLMClient", lambda _settings: SimpleNamespace(complete=respond, close=lambda: None))
    assert main([*resume, "--python", sys.executable, "--message", "Run the check"]) == 0
    assert "[completed] model_stop" in capsys.readouterr().out
    assert len(requests) == len(approvals) == 2
    assert approvals[0][0] == str(workspace.resolve())
    assert approvals[0][1][:4] == [sys.executable, "-I", "-B", "-c"]
    assert approvals[1] == (str(workspace.resolve()), argv)
    replay = SessionReplay(store).replay(session_id, "Replay history.")
    results = [event.payload for event in replay.state.history if isinstance(event.payload, ToolResultPayload)]
    assert len(results) == 1 and results[0].call_id == call.call_id
    assert replay.unknown_call_ids == replay.consumed_call_ids == {call.call_id}
    assert replay.state.current_turn.finished.reason == "model_stop"
