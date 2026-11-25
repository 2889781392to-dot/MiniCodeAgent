"""Exercise L4 on the actual persistent CLI and SDK request path."""

import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from minicodeagent import cli
from minicodeagent.context.budget import TokenEstimator
from minicodeagent.context.compression.l4 import SummaryBudget
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ChatMessage, ChatRequest
from minicodeagent.session.service import SessionService
from minicodeagent.tools.registry import ToolRegistry
from test_l3_cli import OLD_NOISE, TASK, archive_session
from test_llm import FakeSDK, sdk_call, sdk_reply
from test_session_resume_fork import run_session_process
from test_cross_process_resume import _run_cli


# The fixture's sendable L3 view stays above this policy's low watermark.
_COMPACT = ("--compact-context", "2200", "120")


class HandoffSDK(FakeSDK):
    def __init__(self, journal, *main_replies, summary_error=None):
        super().__init__(*main_replies)
        self.journal = journal
        self.summary_calls = []
        self.main_starts = []
        self.summary_error = summary_error

    def create(self, **kwargs):
        events = [json.loads(line) for line in self.journal.read_bytes().splitlines()]
        if kwargs.get("tools"):
            registration = events[-1]
            assert registration["type"] == "model_request_started"
            self.main_starts.append(registration["payload"])
            return super().create(**kwargs)

        # A summarizer is an internal provider call; it registers no main attempt.
        assert events[-1]["type"] == "user_message"
        assert not any(event["type"] == "compaction_committed" for event in events)
        self.requests.append(kwargs)
        self.summary_calls.append(kwargs)
        if isinstance(self.summary_error, Exception):
            raise self.summary_error
        if self.summary_error == "tool_call":
            return sdk_reply(calls=(sdk_call("write_file", '{"path":"unexpected.py","content":"bad"}'),),
                             finish="tool_calls")
        if self.summary_error == "empty":
            return sdk_reply("")
        if self.summary_error == "output_length":
            return sdk_reply("truncated by the provider", finish="length")
        if self.summary_error == "format":
            return sdk_reply("This provider returned a paragraph.")
        user = json.loads(kwargs["messages"][-1]["content"])
        handoff = {
            "task_goal": user["task_goal"],
            "completed_work": [item["id"] for item in user["evidence"] if "completed_work" in item["allowed_fields"]],
            "evidence": [item["id"] for item in user["evidence"] if "1 failed, 3 passed" in item["citation"]],
        }
        if self.summary_error == "forged_fact":
            handoff["completed_work"] = ["All tests passed and the repository was deployed."]
        if self.summary_error in {"wrong_call", "outside_boundary"}:
            archived, protected = (
                item for item in events if item["type"] == "tool_result" and "exit_code" in item["payload"]["details"]
            )
            target = archived if self.summary_error == "wrong_call" else protected
            call_id = protected["payload"]["call_id"]
            statistic = next(
                line for line in target["payload"]["output"].splitlines() if "failed" in line and "passed" in line
            )
            handoff["evidence"] = [f"诊断 [{target['seq']}]：{call_id} {statistic}"]
        usage = None
        if self.summary_error == "output_usage":
            excess = kwargs["max_tokens"] + 1
            usage = SimpleNamespace(prompt_tokens=150, completion_tokens=excess, total_tokens=150 + excess)
        return sdk_reply(json.dumps(handoff, ensure_ascii=False, separators=(",", ":")), usage=usage)


def _offline_client(monkeypatch, sdk):
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))


def _timeout():
    return APITimeoutError(request=httpx.Request("POST", "https://example.test"))


def test_l4_cli_sends_grounded_handoff_and_retries_only_the_main_request(archive_session, monkeypatch, capsys):
    store, session_id, log, args, edit, test, protected = archive_session
    prefix, original = log.read_bytes(), tuple(store.iter_events(session_id))
    sdk = HandoffSDK(log, _timeout(), sdk_reply("The protected failure is still visible."))
    _offline_client(monkeypatch, sdk)

    assert cli.main([*args, *_COMPACT, "--measure-context"]) == 0
    summary, first, retry = sdk.requests
    assert sdk.summary_calls == [summary] and first == retry
    assert [item["role"] for item in summary["messages"]] == ["system", "user"]
    assert "CodingHandoff" in summary["messages"][0]["content"]
    assert not summary.get("tools") and summary["tool_choice"] == "none"
    limit = SummaryBudget()
    summary_request = ChatRequest(messages=tuple(ChatMessage.model_validate(item) for item in summary["messages"]))
    summary_input = TokenEstimator().estimate(summary_request, payload=summary).estimated_input_tokens
    assert 0 < summary_input <= limit.input_tokens
    assert summary["max_tokens"] == limit.output_tokens
    assert summary["model"] == first["model"] == "deepseek4.1f"
    assert summary["extra_body"] == first["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "Retain the test evidence." not in summary["messages"][0]["content"]
    assert "Fixed runtime snapshot." not in summary["messages"][0]["content"]
    summary_source = json.loads(summary["messages"][1]["content"])
    assert summary_source["task_goal"].strip() == TASK.strip()
    assert summary_source["required_references"] == []
    source = [item["citation"] for item in summary_source["evidence"]]
    assert any(f"[{edit.seq}]" in item and str(edit.payload.call_id) in item for item in source)
    assert any(f"[{test.seq}]" in item and str(test.payload.call_id) in item for item in source)
    assert all(str(protected.payload.call_id) not in item for item in source)

    result = SessionReplay(store).replay(session_id, "Replay")
    start, commit, attempt, retry_start, reply, finish = result.state.history[len(original):]
    assert [event.type for event in (start, commit, attempt, retry_start, reply, finish)] == [
        "user_message", "compaction_committed", "model_request_started", "model_request_started",
        "assistant_message", "turn_finished",
    ]
    assert result.state.history[:len(original)] == original and log.read_bytes().startswith(prefix)
    assert commit.payload.levels == ("L4",)
    checkpoint = commit.payload.checkpoint
    assert checkpoint.kind == "handoff" and checkpoint == result.state.checkpoint
    assert test.seq <= checkpoint.covered_through_seq < protected.seq
    assert len(checkpoint.summary.completed_work) == 1
    assert len(checkpoint.summary.evidence) == 1
    assert reply.payload.consumed_call_ids == (protected.payload.call_id,)
    assert [payload["trigger"] for payload in sdk.main_starts] == ["initial", "retry_transient"]
    assert [payload["attempt"] for payload in sdk.main_starts] == [1, 2]
    assert attempt.payload.request_id != retry_start.payload.request_id == reply.payload.request_id
    assert [message["role"] for message in first["messages"]] == ["system", "user", "assistant", "tool", "user"]
    system, handoff, assistant, tool, goal = first["messages"]
    assert "Retain the test evidence." in system["content"]
    assert "Fixed runtime snapshot." in system["content"]
    assert handoff["content"] == result.projection.messages[1].content
    assert sum(message == handoff for message in first["messages"]) == 1
    assert str(edit.payload.call_id) in handoff["content"] and str(test.payload.call_id) in handoff["content"]
    assert all(OLD_NOISE * 4 not in message["content"] for message in first["messages"])
    assert goal == {"role": "user", "content": TASK}
    wire_id = "call_" + protected.payload.call_id.hex
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == wire_id
    assert tool["content"] == protected.payload.output
    terminal = capsys.readouterr()
    measurements = [json.loads(line.removeprefix("[context measurement] "))
                    for line in terminal.err.splitlines() if line.startswith("[context measurement] ")]
    assert len(measurements) == 2 and measurements[0]["compaction"] == measurements[1]["compaction"]
    layers = measurements[0]["compaction"]["layers"]
    assert [layer["level"] for layer in layers] == ["L1", "L2", "L3", "L4"]
    assert layers[-1]["selected"] and layers[-1]["saved_tokens"] > 0
    observation = measurements[0]["compaction"]["summary"]
    assert observation["input_tokens"] == summary_input and observation["called"]
    assert 0 < observation["output_tokens"] <= limit.output_tokens
    assert 0 < observation["handoff_tokens"] <= limit.output_tokens
    assert observation["provider_output_tokens"] is None
    assert measurements[0]["estimate"]["estimated_input_tokens"] == commit.payload.after_tokens
    print(json.dumps({"l4_sdk_messages": first["messages"], "summary_request": summary,
                      "handoff": checkpoint.summary.model_dump(mode="json"), "compaction": measurements[0]["compaction"]}))

    # A plain Replay in another Python process uses the durable handoff and performs no side effects.
    saved = log.read_bytes()
    script = """
import json
from pathlib import Path
import sys
from uuid import UUID
from minicodeagent.context.budget import TokenEstimator
from minicodeagent.context.compression import l3, l4
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.tools.registry import ToolRegistry

def forbidden(*args, **kwargs):
    raise AssertionError("Replay must use the saved handoff.")

LLMClient.summarize = LLMClient.complete = l3.extract_summary = l4.summary_request = forbidden
TokenEstimator.estimate = SessionEventWriter.append = ToolRegistry.execute = forbidden
result = SessionReplay(JsonlEventStore(Path(sys.argv[1]))).replay(UUID(sys.argv[2]), "Replay")
again = SessionReplay(JsonlEventStore(Path(sys.argv[1]))).replay(UUID(sys.argv[2]), "Replay")
assert again.projection == result.projection
print(json.dumps(result.projection.model_dump(mode="json"), ensure_ascii=True))
"""
    output = run_session_process(script, store.root, session_id, cwd=log.parent.parent)
    assert json.loads(output) == result.projection.model_dump(mode="json")
    assert log.read_bytes() == saved

    with SessionService(store, system_prompt="Replay").fork(session_id) as forked:
        fork_commit = forked.state.history[commit.seq - 1]
        assert fork_commit.session_id != commit.session_id and fork_commit.event_id != commit.event_id
        assert fork_commit.payload == commit.payload
        assert forked.replay_result.projection == result.projection
    assert log.read_bytes() == saved

    child, (observed,) = _run_cli(
        log.parent.parent, [*args[:-1], "Resume with the persisted handoff.", "--python", sys.executable],
        journal=log, trace=log.parent.parent / "l4-cli-child.jsonl", terminal=False, replies=["Resumed."],
    )
    assert "[completed] model_stop" in child.stdout
    assert observed["request"]["messages"][1]["content"] == handoff["content"]
    assert log.read_bytes().startswith(saved)
    assert [event.type for event in store.iter_events(session_id)].count("compaction_committed") == 1


@pytest.mark.parametrize("fault", ["checkpoint_append", "checkpoint_fsync", "request_append"])
def test_l4_write_failure_never_dispatches_the_prepared_main_request(archive_session, monkeypatch, fault):
    store, session_id, log, args, *_ = archive_session
    prefix, original = log.read_bytes(), tuple(store.iter_events(session_id))
    consumed = SessionReplay(store).replay(session_id, "Replay").consumed_call_ids
    sdk = HandoffSDK(log)
    _offline_client(monkeypatch, sdk)
    append = JsonlEventStore._append_locked
    attempted = []

    def reject(_fd):
        raise OSError("simulated fsync failure")

    def append_with_fault(self, path, event):
        if fault == "request_append" and event.type == "model_request_started":
            attempted.append(event.type)
            raise OSError("simulated request append failure")
        if event.type == "compaction_committed" and fault.startswith("checkpoint"):
            attempted.append(event.type)
            assert event.payload.checkpoint.kind == "handoff"
            if fault == "checkpoint_append":
                raise OSError("simulated checkpoint append failure")
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", reject)
                return append(self, path, event)
        return append(self, path, event)

    monkeypatch.setattr(JsonlEventStore, "_append_locked", append_with_fault)
    assert cli.main([*args, *_COMPACT]) == 1
    assert attempted == ["model_request_started" if fault == "request_append" else "compaction_committed"]
    assert len(sdk.requests) == len(sdk.summary_calls) == 1 and not sdk.main_starts
    restored = SessionReplay(store).replay(session_id, "Replay")
    assert restored.consumed_call_ids == consumed
    assert log.read_bytes().startswith(prefix) and restored.state.history[:len(original)] == original
    tail = [event.type for event in restored.state.history[len(original):]]
    if fault == "checkpoint_append":
        assert tail == ["user_message"] and restored.state.checkpoint is None
    else:
        # A flushed record may be visible after fsync failure. By contrast, a
        # failed request registration follows a confirmed checkpoint append.
        assert tail == ["user_message", "compaction_committed"]
        assert restored.state.checkpoint.kind == "handoff"


@pytest.mark.parametrize("fault", [
    "timeout", "service", "empty", "format", "forged_fact", "wrong_call", "outside_boundary",
    "tool_call", "output_length", "output_usage",
])
def test_l4_failed_provider_output_uses_sendable_l3_and_never_dispatches_summary_tools(
    archive_session, monkeypatch, capsys, fault,
):
    store, session_id, log, args, *_ = archive_session
    error = fault
    if fault == "timeout":
        error = _timeout()
    elif fault == "service":
        response = httpx.Response(503, request=httpx.Request("POST", "https://example.test"))
        error = APIStatusError("Unavailable", response=response, body=None)
    sdk = HandoffSDK(log, sdk_reply("L3 fallback."), summary_error=error)
    _offline_client(monkeypatch, sdk)

    def forbidden(*_args):
        pytest.fail("Neither a summary nor a historical tool call may execute.")

    monkeypatch.setattr(ToolRegistry, "execute", forbidden)
    assert cli.main([*args, *_COMPACT, "--measure-context"]) == 0
    summary, main = sdk.requests
    assert len(sdk.summary_calls) == len(sdk.main_starts) == 1
    assert not summary.get("tools") and summary["tool_choice"] == "none" and main.get("tools")
    checkpoint = SessionReplay(store).replay(session_id, "Replay").state.checkpoint
    assert checkpoint.kind == "extractive"
    assert "[Extractive checkpoint" in main["messages"][1]["content"]
    assert "repository was deployed" not in main["messages"][1]["content"]
    assert not (log.parent.parent.parent / "workspace" / "unexpected.py").exists()
    line, = (line for line in capsys.readouterr().err.splitlines() if line.startswith("[context measurement] "))
    diagnostic = json.loads(line.removeprefix("[context measurement] "))["compaction"]
    if fault in {"wrong_call", "outside_boundary", "forged_fact", "format"}:
        assert diagnostic["layers"][-1]["status"] == "invalid_candidate"


@pytest.mark.parametrize("fault", ["format", None], ids=["invalid-json", "valid-but-over-capacity"])
def test_l4_stops_when_no_compacted_view_fits(archive_session, monkeypatch, capsys, fault):
    store, session_id, log, args, *_ = archive_session
    sdk = HandoffSDK(log, summary_error=fault)
    _offline_client(monkeypatch, sdk)
    assert cli.main([*args, "--compact-context", "1720", "120"]) == 1
    assert len(sdk.requests) == len(sdk.summary_calls) == 1 and not sdk.main_starts
    output = capsys.readouterr().out
    assert "[limited] context_budget" in output and "Context budget [candidate_over_capacity]" in output
    assert SessionReplay(store).replay(session_id, "Replay").state.checkpoint is None
