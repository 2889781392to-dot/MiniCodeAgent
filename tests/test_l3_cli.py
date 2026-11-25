"""Exercise extractive checkpoints through the persistent CLI and model boundary."""

import json
import os
import sys
from uuid import UUID, uuid4

import httpx
import pytest
from openai import APITimeoutError

from minicodeagent import cli
from minicodeagent.config import Settings
from minicodeagent.context import manager
from minicodeagent.context.events import (
    AssistantMessagePayload, ModelRequestStartedPayload, ToolResultPayload, TurnFinishedPayload, UserMessagePayload,
)
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionService
from test_l2_cli import _command_result
from test_llm import FakeSDK, sdk_call, sdk_reply, settings
from test_cross_process_resume import _run_cli


TASK = "\n\nInspect the new task goal without removing this whitespace.\n\n"
OLD_NOISE = "Supplementary earlier discussion was included for completeness.\n"


def _call(name, number, **arguments):
    return ToolCall(name=name, provider_call_id=f"fixture-{number}", call_id=UUID(int=number),
                    arguments_raw=json.dumps(arguments))


@pytest.fixture
def archive_session(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("value = 42\n", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("Retain the test evidence.", encoding="utf-8")
    store = JsonlEventStore(tmp_path / "data")
    edited = _call("edit_file", 21, path="totals.py", old_text="return 0", new_text="return total")
    tested = _call("run_command", 22, argv=["python", "-m", "pytest", "-q"])
    protected = _call("run_command", 23, argv=["python", "-m", "pytest", "-q"])
    with SessionService(store).create(workspace) as handle:
        session_id = handle.session_id
        writer = handle.writer

        def request(turn, attempt):
            payload = ModelRequestStartedPayload(
                turn_id=turn, request_id=uuid4(), attempt=attempt, trigger="initial" if attempt == 1 else "after_tools",
            )
            writer.append("model_request_started", payload)
            return payload.request_id

        def batch(task, calls):
            turn = uuid4()
            writer.append("user_message", UserMessagePayload(turn_id=turn, content=task))
            started = request(turn, 1)
            writer.append("assistant_message", AssistantMessagePayload(
                turn_id=turn, request_id=started, content="", finish_reason="tool_calls", tool_calls=calls,
            ))
            return turn

        older = batch("\n" * 700 + "Fix the signed totals and preserve the tests.\n" + OLD_NOISE * 170,
                      (edited, tested))
        edit = writer.append("tool_result", ToolResultPayload(
            turn_id=older, call_id=edited.call_id, status="ok", output="Edited totals.py.",
            details={"path": "totals.py"},
        ))
        test = writer.append("tool_result", _command_result(older, tested, long=True))
        reply = writer.append("assistant_message", AssistantMessagePayload(
            turn_id=older, request_id=request(older, 2), finish_reason="stop",
            consumed_call_ids=(edited.call_id, tested.call_id),
            content="The signed totals assertion still fails.\nNext: check the edge case.\n" + OLD_NOISE * 80,
        ))
        writer.append("turn_finished", TurnFinishedPayload(
            turn_id=older, status="completed", reason="model_stop", assistant_seq=reply.seq,
        ))
        later = batch("Preserve these results because they were not consumed.", (protected,))
        protected_result = writer.append("tool_result", _command_result(later, protected, long=False))
        request(later, 2)
        writer.append("turn_finished", TurnFinishedPayload(turn_id=later, status="failed", reason="timeout"))
    configured = settings(thinking=True)
    monkeypatch.setattr(Settings, "from_env", lambda _env: configured)
    monkeypatch.setattr(cli, "_run_environment_prompt", lambda *_args: lambda: "Fixed runtime snapshot.")
    log = store.root / "sessions" / f"{session_id}.jsonl"
    args = ["resume", str(session_id), "--data-root", str(store.root), "--message", TASK]
    return store, session_id, log, args, edit, test, protected_result


def test_l3_checkpoint_reaches_fake_sdk_with_only_the_protected_tail(archive_session, monkeypatch, capsys):
    store, session_id, log, args, edit, test, protected = archive_session
    before = log.read_bytes()
    records = tuple(store.iter_events(session_id))
    attempts = []

    class SDK(FakeSDK):
        def create(self, **kwargs):
            journal = [json.loads(line) for line in log.read_bytes().splitlines()]
            assert journal[-1]["type"] == "model_request_started"
            commit, = [event for event in journal if event["type"] == "compaction_committed"]
            assert commit["seq"] < journal[-1]["seq"]
            attempts.append(journal[-1]["payload"])
            assert commit["payload"]["checkpoint"]["kind"] == "extractive"
            return super().create(**kwargs)

    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    sdk = SDK(timeout, sdk_reply(calls=(sdk_call(),), finish="tool_calls"), sdk_reply("finished"))
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    assert cli.main([*args, "--compact-context", "3800", "150", "--measure-context"]) == 0
    first, retry, final = sdk.requests
    assert first == retry
    assert first["model"] == "deepseek4.1f" and first["extra_body"] == {"thinking": {"type": "disabled"}}
    assert first["tools"] == final["tools"]
    assert [start["attempt"] for start in attempts] == [1, 2, 3]
    assert [start["trigger"] for start in attempts] == ["initial", "retry_transient", "after_tools"]
    assert len({start["request_id"] for start in attempts}) == 3
    assert "Retain the test evidence." in first["messages"][0]["content"]
    assert "Fixed runtime snapshot." in first["messages"][0]["content"]
    assert first["messages"][0] == final["messages"][0]
    assert first["messages"][-1] == {"role": "user", "content": TASK}
    assert all(OLD_NOISE * 4 not in m["content"] for m in first["messages"])

    result = SessionReplay(store).replay(session_id, "Replay")
    assert log.read_bytes().startswith(before) and result.state.history[:len(records)] == records
    committed, = [e.payload for e in result.state.history if e.type == "compaction_committed"]
    cp = result.state.checkpoint
    assert cp == committed.checkpoint and cp.covered_through_seq < protected.seq
    assert cp.covered_through_seq >= test.seq
    projected_summary = result.projection.messages[1].content
    assert first["messages"][1]["role"] == "user"
    assert first["messages"][1]["content"] == final["messages"][1]["content"] == projected_summary
    assert sum(message["content"] == projected_summary for message in first["messages"]) == 1
    assert "totals.py" in projected_summary and "1 failed" in projected_summary
    assert "Next: check the edge case." in projected_summary
    for event in (edit, test):
        assert str(event.payload.call_id) in projected_summary and str(event.seq) in projected_summary

    first_ids = [m["tool_call_id"] for m in first["messages"] if m["role"] == "tool"]
    assert first_ids == ["call_" + protected.payload.call_id.hex]
    assert [item["id"] for m in first["messages"] for item in m.get("tool_calls", [])] == first_ids
    assert next(m for m in first["messages"] if m["role"] == "tool")["content"] == protected.payload.output
    replies = [e.payload for e in result.state.history[len(records):] if e.type == "assistant_message"]
    assert replies[0].consumed_call_ids == (protected.payload.call_id,)
    assert replies[1].consumed_call_ids == (protected.payload.call_id, replies[0].tool_calls[0].call_id)

    captured = [json.loads(line.removeprefix("[context measurement] "))
                for line in capsys.readouterr().err.splitlines() if line.startswith("[context measurement] ")]
    assert captured[0]["compaction"] == captured[1]["compaction"]
    layers = captured[0]["compaction"]["layers"]
    assert [layer["level"] for layer in layers] == ["L1", "L2", "L3", "L4"]
    assert layers[-1]["status"] == "not_needed" and not layers[-1]["triggered"]
    assert captured[0]["estimate"]["estimated_input_tokens"] == committed.after_tokens < committed.before_tokens
    prefix = log.read_bytes()
    child, (observed,) = _run_cli(
        log.parent.parent, [*args[:-1], "Resume after restart.", "--python", sys.executable],
        journal=log, trace=log.parent.parent / "l3-child.jsonl", terminal=False, replies=["Resumed."],
    )
    assert "[completed] model_stop" in child.stdout
    assert observed["request"]["messages"][1:-1] == [
        m.model_dump(mode="json") for m in result.projection.messages[1:]
    ]
    assert log.read_bytes().startswith(prefix)
    assert [e.type for e in store.iter_events(session_id)].count("compaction_committed") == 1
    print(json.dumps({
        "comparison": captured[0]["compaction"], "checkpoint": cp.model_dump(mode="json"),
        "protected_tool_ids": first_ids, "latest_user": first["messages"][-1]["content"],
        "sdk_roles": [m["role"] for m in first["messages"]],
    }))


@pytest.mark.parametrize("fault", ["checkpoint_append", "checkpoint_fsync", "request_append"])
def test_l3_submission_failure_stops_dispatch_and_preserves_history(archive_session, monkeypatch, fault):
    store, session_id, log, args, *_ = archive_session
    before = log.read_bytes()
    original = tuple(store.iter_events(session_id))
    sdk = FakeSDK()
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    append = JsonlEventStore._append_locked
    failures = []

    def reject(_fd):
        raise OSError("simulated fsync failure")

    def append_with_fault(self, path, event):
        if fault == "request_append" and event.type == "model_request_started":
            failures.append(event.type)
            raise OSError("simulated request append failure")
        if event.type == "compaction_committed" and fault.startswith("checkpoint"):
            failures.append(event.type)
            assert event.payload.checkpoint is not None
            if fault == "checkpoint_append":
                raise OSError("simulated checkpoint append failure")
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", reject)
                return append(self, path, event)
        return append(self, path, event)

    monkeypatch.setattr(JsonlEventStore, "_append_locked", append_with_fault)
    assert cli.main([*args, "--compact-context", "3800", "150"]) == 1
    assert failures == ["model_request_started" if fault == "request_append" else "compaction_committed"]
    assert not sdk.requests and log.read_bytes().startswith(before)
    restored = SessionReplay(store).replay(session_id, "Replay")
    assert restored.state.history[:len(original)] == original
    tail = [event.type for event in restored.state.history[len(original):]]
    if fault == "checkpoint_append":
        assert tail == ["user_message"] and restored.state.checkpoint is None
    else:
        # The flushed fsync-failed event may be visible on reread.
        assert tail == ["user_message", "compaction_committed"] and restored.state.checkpoint is not None


@pytest.mark.parametrize("fault", ["extraction_failed", "invalid_candidate", "no_gain", "candidate_over_capacity"])
def test_l3_rejection_uses_only_a_sendable_earlier_view(archive_session, monkeypatch, capsys, fault):
    store, session_id, log, args, *_ = archive_session
    before = log.read_bytes()
    over_capacity = fault == "candidate_over_capacity"
    sdk = FakeSDK() if over_capacity else FakeSDK(sdk_reply("finished with L1/L2"))
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    estimates = iter((1800, 1400, 1300, 1200) if over_capacity else (1800, 1400, 1050, 1050))
    monkeypatch.setattr(manager.ContextWindowManager, "_tokens", lambda *_args: next(estimates))
    monkeypatch.delattr(LLMClient, "summarize")
    if fault == "extraction_failed":
        def fail(_history):
            raise RuntimeError("simulated extraction failure")

        monkeypatch.setattr(manager, "l3_checkpoint", fail)
    elif fault == "invalid_candidate":
        propose = manager.l3_checkpoint

        def invalid(history):
            checkpoint = propose(history)
            assert checkpoint is not None
            return checkpoint.model_copy(update={"covered_through_seq": history[-1].seq})

        monkeypatch.setattr(manager, "l3_checkpoint", invalid)

    assert cli.main([*args, "--compact-context", "1200", "0", "--measure-context"]) == int(over_capacity)
    assert log.read_bytes().startswith(before)
    events = tuple(store.iter_events(session_id))
    commits = [e.payload for e in events if e.type == "compaction_committed"]
    terminal = capsys.readouterr()
    if over_capacity:
        assert not sdk.requests and not commits and "context_budget" in terminal.out
        return
    committed, = commits
    assert committed.levels == ("L1", "L2") and committed.checkpoint is None
    request, = sdk.requests
    assert sum(message["role"] == "system" for message in request["messages"]) == 1
    assert any("[L2 tool result excerpt]" in message["content"] for message in request["messages"])
    line, = [s for s in terminal.err.splitlines() if s.startswith("[context measurement] ")]
    diag = json.loads(line.removeprefix("[context measurement] "))["compaction"]
    assert diag["layers"][-1]["status"] == fault and not diag["layers"][-1]["selected"]
