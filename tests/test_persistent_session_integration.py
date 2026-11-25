"""V0.2 acceptance across the CLI, disk journal, and real coding tools."""

from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from uuid import UUID

import pytest

from minicodeagent import cli
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.protocol import ModelReply, ToolCall
from minicodeagent.tools.process import MAX_STREAM_BYTES
from minicodeagent.tools.registry import ToolRegistry
from test_python_environment_cli import capture_model, observation, resume


def test_persistent_cli_repairs_project_and_replays_actual_tool_results(tmp_path, monkeypatch, capsys):
    workspace = (tmp_path / "sample project").resolve()
    workspace.mkdir()
    data_root = tmp_path / "data"
    fixture = Path(__file__).resolve().parents[1] / "examples" / "bugfix"
    source = workspace / "totals.py"
    acceptance = workspace / "test_totals.py"
    for path in (source, acceptance):
        copyfile(fixture / path.name, path)
    original_test_hash = sha256(acceptance.read_bytes()).hexdigest()
    rules = workspace / "AGENTS.md"
    rules.write_text("Keep the acceptance tests unchanged. Use the selected Python to run Pytest.", encoding="utf-8")
    rules_bytes = rules.read_bytes()
    python = str(Path(sys.executable).absolute())
    test_argv = [
        python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--basetemp", str(tmp_path / "nested pytest"),
    ]
    task = "Fix total_even in totals.py without changing the tests."

    assert cli.main(["new", "--data-root", str(data_root), "--workspace", str(workspace)]) == 0
    session_id = UUID(capsys.readouterr().out.strip())
    log = data_root / "sessions" / f"{session_id}.jsonl"
    assert [json.loads(row)["type"] for row in log.read_bytes().splitlines()] == ["session_created"]

    def call(name, **arguments):
        return ToolCall(
            provider_call_id=f"acceptance-{name}-{len(batches)}",
            name=name, arguments_raw=json.dumps(arguments),
        )

    batches = []
    starts = []
    command_feedback = []

    def respond(request, count):
        # Peek at the real journal before producing any reply, while the CLI holds its write lease.
        start = json.loads(log.read_bytes().splitlines()[-1])
        assert start["type"] == "model_request_started"
        assert start["payload"]["attempt"] == count
        starts.append(start)
        assert request.messages[0].content.count(rules_bytes.decode()) == 1
        env = observation(request)
        assert (env["workspace"], env["python_executable"], env["source"]) == (str(workspace), python, "explicit")
        assert (env["verification"], env["pytest_status"]) == ("verified", "discoverable")

        if batches:
            previous = batches[-1]
            assistant, *results = request.messages[-(len(previous) + 1):]
            assert assistant.role == "assistant" and assistant.tool_calls == previous
            assert [message.role for message in results] == ["tool"] * len(previous)
            assert [message.call_id for message in results] == [item.call_id for item in previous]
        if count == 1:
            batch = (call("list_files"), call("read_file", path="totals.py"))
        elif count == 2:
            assert all(path.name in results[0].content for path in (rules, source, acceptance))
            assert "return 0" in results[1].content
            batch = (call("edit_file", path="totals.py", old_text="return 0", new_text="return sum(values)"),)
        elif count == 3:
            assert "Edited totals.py." in results[0].content
            assert "return sum(values)" in source.read_text(encoding="utf-8")
            batch = (call("run_command", argv=test_argv),)
        elif count == 4:
            assert "Exit code: 1" in results[0].content
            assert "assert 9 == 6" in results[0].content and "1 failed, 1 passed" in results[0].content
            command_feedback.append(results[0].content)
            batch = (call("edit_file", path="totals.py", old_text="return sum(values)",
                          new_text="return sum(value for value in values if value % 2 == 0)"),)
        elif count == 5:
            assert "Edited totals.py." in results[0].content
            batch = (call("run_command", argv=test_argv),)
        else:
            assert count == 6 and "Exit code: 0" in results[0].content and "2 passed" in results[0].content
            command_feedback.append(results[0].content)
            return ModelReply(content="Fixed totals.py and verified both tests.", finish_reason="stop")
        batches.append(batch)
        return ModelReply(finish_reason="tool_calls", tool_calls=batch)

    class TerminalInput(StringIO):
        def isatty(self):
            return True

    terminal = TerminalInput("y\ny\ny\n")
    monkeypatch.setattr(sys, "stdin", terminal)
    requests = capture_model(monkeypatch, respond)
    spawns = []
    popen = subprocess.Popen

    def record_spawn(argv, **options):
        spawns.append((list(argv), str(options["cwd"])))
        assert options["shell"] is False
        return popen(argv, **options)

    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    assert resume(data_root, session_id, "--python", python, "--message", task) == 0
    stdout, stderr = capsys.readouterr()
    assert "[completed] model_stop" in stdout and "Fixed totals.py and verified both tests." in stdout
    assert stdout.count("Run this command once? [y/N]") == 3
    approved_argv = [json.loads(line.removeprefix("Command argv: ")) for line in stderr.splitlines()
                     if line.startswith("Command argv: ")]
    approved_cwd = [json.loads(line.removeprefix("Command cwd: ")) for line in stderr.splitlines()
                    if line.startswith("Command cwd: ")]
    assert spawns == list(zip(approved_argv, approved_cwd))
    assert len(spawns) == 3 and approved_argv[0][:4] == [python, "-I", "-B", "-c"]
    assert approved_cwd == [str(workspace)] * 3
    assert approved_argv[1:] == [test_argv, test_argv]
    assert source.read_text(encoding="utf-8") == (
        "def total_even(values):\n"
        "    return sum(value for value in values if value % 2 == 0)\n"
    )
    assert sha256(acceptance.read_bytes()).hexdigest() == original_test_hash
    assert rules.read_bytes() == rules_bytes

    store = JsonlEventStore(data_root)
    events = tuple(store.iter_events(session_id))
    replies = [event for event in events if event.type == "assistant_message"]
    assert len(requests) == len(replies) == len(starts) == 6
    assert len({start["payload"]["request_id"] for start in starts}) == 6
    assert [start["payload"]["trigger"] for start in starts] == ["initial"] + ["after_tools"] * 5
    for start, reply in zip(starts, replies):
        assert (str(reply.payload.request_id), str(reply.payload.turn_id), reply.seq) == (
            start["payload"]["request_id"], start["payload"]["turn_id"], start["seq"] + 1,
        )
    assert [reply.payload.tool_calls for reply in replies] == [*batches, ()]
    calls = [item for batch in batches for item in batch]
    outcomes = [event.payload for event in events if event.type == "tool_result"]
    assert [result.call_id for result in outcomes] == [item.call_id for item in calls]
    assert [result.status for result in outcomes] == ["ok", "ok", "ok", "error", "ok", "ok"]
    commands = outcomes[3], outcomes[5]
    assert [result.details["exit_code"] for result in commands] == [1, 0]
    assert [result.output for result in commands] == command_feedback
    for result in commands:
        for stream in ("stdout", "stderr"):
            assert not result.details[f"{stream}_truncated"]
            assert len(result.details[stream].encode()) <= MAX_STREAM_BYTES
        assert not result.details["capture_incomplete"]

    frozen_log = log.read_bytes()
    frozen_files = {path.relative_to(workspace): path.read_bytes() for path in workspace.rglob("*") if path.is_file()}

    def unexpected(*_args, **_kwargs):
        pytest.fail("Replay must not execute tools, start a process, call a model, or append events")

    monkeypatch.setattr(cli, "LLMClient", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(ToolRegistry, "execute", unexpected)
    monkeypatch.setattr(SessionEventWriter, "append", unexpected)
    replay = SessionReplay(store).replay(session_id, cli.SYSTEM_PROMPT)
    assert SessionReplay(JsonlEventStore(data_root)).replay(session_id, cli.SYSTEM_PROMPT) == replay
    assert replay.state.history == events
    assert replay.state.current_turn.latest_reply == replies[-1]
    finished = replay.state.current_turn.finished
    assert (finished.status, finished.reason, finished.assistant_seq) == ("completed", "model_stop", replies[-1].seq)
    assert replay.state.current_turn.model_attempts == 6
    assert replay.consumed_call_ids == frozenset(item.call_id for item in calls)
    assert not replay.pending_calls and not replay.unknown_call_ids and not replay.diagnostics
    assert replay.last_seq == events[-1].seq and replay.next_seq == events[-1].seq + 1
    assert replay.projection.messages[-1].content == "Fixed totals.py and verified both tests."
    assert all("Current run environment observation" not in msg.content for msg in replay.projection.messages)
    assert log.read_bytes() == frozen_log
    assert {path.relative_to(workspace): path.read_bytes() for path in workspace.rglob("*") if path.is_file()} == frozen_files

    print(json.dumps({
        "session_id": str(session_id), "last_seq": replay.last_seq, "test_sha256": original_test_hash,
        "first_request": {
            "request_id": starts[0]["payload"]["request_id"],
            "roles": [message.role for message in requests[0].messages],
            "user": requests[0].messages[-1].content,
            "project_background": rules_bytes.decode(),
            "environment": observation(requests[0]),
            "tools": [spec.name for spec in requests[0].tools],
        },
        "edit_and_test_calls": [str(item.call_id) for item in calls[2:]],
        "approved_test_argv": approved_argv[1],
        "pytest_exit_codes": [result.details["exit_code"] for result in commands],
        "pytest_summaries": [result.details["stdout"].splitlines()[-1] for result in commands],
    }))
