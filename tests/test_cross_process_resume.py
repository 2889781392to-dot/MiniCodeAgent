"""Exercise persisted CLI turns across process exits and recovery."""

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID

from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore


# Keep the production CLI, journal, project loaders and tools in each child.
# Script the model and supply answers to its real confirmation prompts.
_OFFLINE_CLI = """
import json
import os
import sys
from pathlib import Path

from minicodeagent import cli
from minicodeagent.config import LoopLimits
from minicodeagent.protocol import ModelError, ModelReply, ToolCall
from minicodeagent.tools.registry import ToolRegistry

journal = Path(sys.argv[1])
trace = Path(sys.argv[2])
options = json.loads(sys.argv[3])
if options["terminal"]:
    sys.stdin.isatty = lambda: True

def record(**entry):
    with trace.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\\n")

class FakeModel:
    def __init__(self, _settings):
        self.replies = iter(options["replies"])

    def complete(self, request):
        started = json.loads(journal.read_bytes().splitlines()[-1])
        assert started["type"] == "model_request_started"
        record(pid=os.getpid(), started=started, request=request.model_dump(mode="json"))
        reply = next(self.replies)
        if reply is None:
            raise ModelError("timeout", "Synthetic transient timeout.")
        if isinstance(reply, str):
            return ModelReply(content=reply, finish_reason="stop")
        calls = tuple(ToolCall(
            provider_call_id=call_id, name=name, arguments_raw=json.dumps(arguments),
        ) for call_id, name, arguments in reply)
        return ModelReply(tool_calls=calls, finish_reason="tool_calls")

    def close(self):
        pass

execute = ToolRegistry.execute
def track_dispatch(self, call, workspace):
    if call.provider_call_id == options["crash_before"]:
        os._exit(73)
    assert call.provider_call_id in options["dispatch"], "Unexpected tool dispatch"
    record(dispatched=call.model_dump(mode="json"))
    return execute(self, call, workspace)

if options["provider_limit"] is not None:
    read_settings = cli.Settings.from_env
    def read_limited_settings(environ):
        limits = LoopLimits(max_provider_calls=options["provider_limit"])
        return read_settings(environ).model_copy(update={"limits": limits})
    cli.Settings.from_env = read_limited_settings

cli.LLMClient = FakeModel
ToolRegistry.execute = track_dispatch
sys.exit(cli.main(sys.argv[4:]))
"""

_SYSTEM_KEYS = (
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP",
    "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "SYSTEMDRIVE",
)


def _run_cli(tmp_path, argv, *, code=0, stdin="", journal=None, trace=None, **options):
    env = {key: os.environ[key] for key in _SYSTEM_KEYS if key in os.environ}
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONIOENCODING="utf-8",
        MINICODEAGENT_API_KEY="synthetic-offline-resume-key",
        MINICODEAGENT_MODEL="fake-offline-resume-model",
    )
    command = [sys.executable, "-B"]
    if journal is None:
        command.extend(["-m", "minicodeagent"])
    else:
        settings = dict(replies=[], dispatch=[], crash_before=None, provider_limit=None, terminal=True)
        settings.update(options)
        command.extend(["-c", _OFFLINE_CLI, str(journal), str(trace), json.dumps(settings)])
    result = subprocess.run(
        [*command, *map(str, argv)], cwd=tmp_path, input=stdin, encoding="utf-8",
        capture_output=True, timeout=20, env=env,
    )
    assert result.returncode == code, (result.returncode, result.stdout, result.stderr)
    entries = []
    if trace is not None and trace.exists():
        entries = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    return result, entries


def _new_session(tmp_path):
    workspace = tmp_path / "Python project"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("Use the original project rule.", encoding="utf-8")
    data = tmp_path / "data"
    result, _ = _run_cli(tmp_path, ["new", "--data-root", data, "--workspace", workspace])
    session = UUID(result.stdout.strip())
    journal = data / "sessions" / f"{session}.jsonl"
    resume = ["resume", session, "--data-root", data, "--python", sys.executable]
    return workspace.resolve(), data, session, journal, resume


def _assert_run_context(request, workspace, rule):
    system = request["messages"][0]
    assert system["role"] == "system"
    assert rule in system["content"]
    marker = "[Current run environment observation]\n"
    assert system["content"].count(marker) == 1
    observation, _ = json.JSONDecoder().raw_decode(system["content"].split(marker)[1])
    assert observation["workspace"] == str(workspace)
    assert observation["python_executable"] == sys.executable
    assert observation["source"] == "explicit" and observation["verification"] == "verified"
    assert observation["pytest_status"] == "discoverable"


def test_resume_in_another_process_reads_saved_tool_history_and_starts_new_turn(tmp_path):
    workspace, data, session, journal, resume = _new_session(tmp_path)
    note = workspace / "note.txt"
    note.write_text("prior tool evidence", encoding="utf-8")
    result, first = _run_cli(
        tmp_path, [*resume, "--message", "Read the note."], journal=journal,
        trace=tmp_path / "first.jsonl", stdin="yes\n",
        replies=[[["initial-read", "read_file", {"path": note.name}]], "Read the note."],
        dispatch=["initial-read"],
    )
    assert "[completed] model_stop" in result.stdout
    first_requests = [entry for entry in first if "request" in entry]
    assert [entry["started"]["payload"]["attempt"] for entry in first_requests] == [1, 2]
    assert first[1]["dispatched"]["provider_call_id"] == "initial-read"
    previous_tool = first[-1]["request"]["messages"][-1]
    assert previous_tool["role"] == "tool" and "prior tool evidence" in previous_tool["content"]
    original = journal.read_bytes()
    (workspace / "AGENTS.md").write_text("Use the revised project rule.", encoding="utf-8")
    assert journal.read_bytes() == original

    result, (second,) = _run_cli(
        tmp_path, [*resume, "--message", "Recall the note."], journal=journal,
        trace=tmp_path / "second.jsonl", stdin="yes\n", replies=["Recalled the note."],
    )
    assert "[completed] model_stop" in result.stdout
    assert first_requests[0]["pid"] != second["pid"]
    start = second["started"]["payload"]
    earlier_start = first_requests[-1]["started"]["payload"]
    assert (start["attempt"], start["trigger"]) == (1, "initial")
    assert start["turn_id"] != earlier_start["turn_id"]
    _assert_run_context(second["request"], workspace, "Use the revised project rule.")
    _assert_run_context(first_requests[0]["request"], workspace, "Use the original project rule.")
    messages = second["request"]["messages"]
    assert [message["content"] for message in messages if message["role"] == "user"] == [
        "Read the note.", "Recall the note.",
    ]
    assert [message for message in messages if message["role"] == "tool"] == [previous_tool]
    assert journal.read_bytes().startswith(original)
    replay = SessionReplay(JsonlEventStore(data)).replay(session, "fixed prompt")
    assert replay.state.current_turn.finished.reason == "model_stop"
    assert replay.consumed_call_ids == {UUID(previous_tool["call_id"])}
    assert replay.pending_calls == replay.diagnostics == ()
    assert note.read_bytes() == b"prior tool evidence"


def test_cross_process_resume_settles_unknown_in_order_and_keeps_request_budget(tmp_path):
    workspace, data, session, journal, resume = _new_session(tmp_path)
    writes = [
        ["already-saved", "write_file", {"path": "already.txt", "content": "Persisted tool effect."}],
        ["pending-first", "write_file", {"path": "first.txt", "content": "Must not execute."}],
        ["pending-second", "write_file", {"path": "second.txt", "content": "Must not execute."}],
    ]
    _, initial = _run_cli(
        tmp_path, [*resume, "--message", "Write the fixture."], code=73, journal=journal,
        trace=tmp_path / "abrupt.jsonl", stdin="yes\n", provider_limit=3,
        replies=[None, writes], dispatch=["already-saved"], crash_before="pending-first",
    )
    starts = [entry["started"]["payload"] for entry in initial if "started" in entry]
    assert [(entry["attempt"], entry["trigger"]) for entry in starts] == [
        (1, "initial"), (2, "retry_transient"),
    ]
    before = journal.read_bytes()
    replay = SessionReplay(JsonlEventStore(data))
    incomplete = replay.replay(session, "fixed prompt")
    pending = [call.call_id for call in incomplete.pending_calls]
    assert [call.provider_call_id for call in incomplete.pending_calls] == ["pending-first", "pending-second"]
    assert incomplete.projection is None and incomplete.unknown_call_ids == set(pending)
    assert [entry["dispatched"]["provider_call_id"] for entry in initial if "dispatched" in entry] == [
        "already-saved",
    ]
    assert (workspace / "already.txt").read_bytes() == b"Persisted tool effect."

    for terminal, answer in ((False, ""), (True, "no\n")):
        rejected, activity = _run_cli(
            tmp_path, [*resume, "--continue"], code=2, stdin=answer, journal=journal,
            trace=tmp_path / "refusal.jsonl", terminal=terminal,
        )
        assert "Confirmation required for pending tool calls" in rejected.stderr
        assert rejected.stderr.index(f"[unknown {pending[0]}]") < rejected.stderr.index(f"[unknown {pending[1]}]")
        assert "Command argv:" not in rejected.stderr
        assert activity == [] and journal.read_bytes() == before

    result, (resumed,) = _run_cli(
        tmp_path, [*resume, "--continue"], code=1, journal=journal,
        trace=tmp_path / "confirmed.jsonl", stdin="yes\nyes\n", provider_limit=3,
        replies=[[["fresh-list", "list_files", {}]]],
    )
    assert "[limited] provider_call_limit" in result.stdout
    info = resumed["started"]["payload"]
    assert (info["attempt"], info["trigger"], info["turn_id"]) == (3, "resume", starts[1]["turn_id"])
    _assert_run_context(resumed["request"], workspace, "Use the original project rule.")
    history = [message for message in resumed["request"]["messages"] if message["role"] == "tool"]
    assert [UUID(message["call_id"]) for message in history] == [
        UUID(initial[-1]["dispatched"]["call_id"]), *pending,
    ]
    assert all("unknown" in message["content"] for message in history[1:])
    assert not any((workspace / name).exists() for name in ("first.txt", "second.txt"))
    assert (workspace / "already.txt").read_bytes() == b"Persisted tool effect."
    assert journal.read_bytes().startswith(before)

    recovered = replay.replay(session, "fixed prompt")
    appended = recovered.state.history[len(incomplete.state.history):]
    assert [(event.payload.call_id, event.payload.status, event.payload.details) for event in appended[:2]] == [
        (call_id, "interrupted", {"execution_outcome": "unknown"}) for call_id in pending
    ]
    assert [event.type for event in appended[2:]] == [
        "model_request_started", "assistant_message", "turn_finished",
    ]
    reply = appended[3].payload
    assert reply.request_id == UUID(info["request_id"])
    assert reply.tool_calls == () and [call.provider_call_id for call in reply.rejected_tool_calls] == ["fresh-list"]
    assert recovered.consumed_call_ids == {UUID(message["call_id"]) for message in history}
    assert recovered.state.current_turn.model_attempts == 3 and recovered.state.current_turn.retried_transient
    assert recovered.state.current_turn.finished.reason == "provider_call_limit"
    assert recovered.pending_calls == recovered.diagnostics == ()
    assert recovered.unknown_call_ids == set(pending)
