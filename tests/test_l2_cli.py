"""Verify layered compaction on the actual persistent CLI request path."""

from copy import deepcopy
import json
import sys
from uuid import UUID, uuid4

import httpx
import pytest
from openai import APITimeoutError

from minicodeagent import agent, cli
from minicodeagent.config import Settings
from minicodeagent.context import manager
from minicodeagent.context.events import (
    AssistantMessagePayload, ModelRequestStartedPayload, TextReplacement, ToolResultPayload, TurnFinishedPayload,
    UserMessagePayload,
)
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ToolCall
from minicodeagent.session.service import SessionService
from test_llm import FakeSDK, sdk_call, sdk_reply, settings
from test_cross_process_resume import _run_cli


LATEST = "\n\nInspect the newest result and keep this goal.\n\n"
OLD = "\n" * 1300 + "Verify the total and keep its evidence." + "\n" * 1300
HEAD = "================ test session starts ================"
FAILURE = "tests/test_totals.py:17: AssertionError: assert 9 == 6"
TAIL = "1 failed, 3 passed in 0.40s"


def _call(name, number, **arguments):
    return ToolCall(
        provider_call_id=f"saved-{number}", call_id=UUID(int=number),
        name=name, arguments_raw=json.dumps(arguments),
    )


def _command_result(turn, call, *, long, unknown=False):
    noise = "collector repeated the same routine progress record\n" * (112 if long else 1)
    stdout = f"{HEAD}\n{noise}{FAILURE}\n{noise[:180]}\n{TAIL}\n"
    stderr = "RuntimeError: subprocess reported failed assertions\n"
    header = "Command timed out after 1 seconds." if unknown else "Exit code: 1"
    details = {
        "exit_code": None if unknown else 1, "stdout": stdout, "stderr": stderr,
        "stdout_truncated": False, "stderr_truncated": False, "capture_incomplete": False,
        "error_code": "command_timeout" if unknown else "command_failed",
    }
    if unknown:
        details["execution_outcome"] = "unknown"
    return ToolResultPayload(
        turn_id=turn, call_id=call.call_id, status="timeout" if unknown else "error",
        output=f"{header}\nstdout:\n{stdout}\nstderr:\n{stderr}", details=details,
    )


@pytest.fixture
def history_session(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("value = 42\n", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("Keep failure evidence.", encoding="utf-8")
    store = JsonlEventStore(tmp_path / "data")
    command = ["python", "-m", "pytest", "-q"]
    old_calls = (_call("run_command", 10, argv=command), _call("search_code", 11, query="total", path="."))
    unread = _call("run_command", 12, argv=command)
    unknown = _call("run_command", 13, argv=command)
    recent = _call("run_command", 14, argv=command)
    with SessionService(store).create(workspace) as handle:
        session_id = handle.session_id
        writer = handle.writer

        def request(turn, attempt):
            start = ModelRequestStartedPayload(
                turn_id=turn, request_id=uuid4(), attempt=attempt, trigger="initial" if attempt == 1 else "after_tools",
            )
            writer.append("model_request_started", start)
            return start.request_id

        def turn_with_batch(calls, *, task, consumed=None, is_unknown=False):
            turn = uuid4()
            writer.append("user_message", UserMessagePayload(turn_id=turn, content=task))
            writer.append("assistant_message", AssistantMessagePayload(
                turn_id=turn, request_id=request(turn, 1), content="", finish_reason="tool_calls", tool_calls=calls,
            ))
            result_seqs = []
            for call in calls:
                if call.name == "search_code":
                    matches = [
                        f"pkg/{file}:{line}: total = value + offset"
                        for file in ("totals.py", "summary.py") for line in range(1, 21)
                    ]
                    payload = ToolResultPayload(
                        turn_id=turn, call_id=call.call_id, status="ok", output="\n".join(matches),
                        details={"count": len(matches), "result_limit_reached": False,
                                 "scan_byte_limit_reached": False, "scan_entry_limit": 10000},
                    )
                else:
                    payload = _command_result(turn, call, long=call == old_calls[0], unknown=is_unknown)
                result_seqs.append(writer.append("tool_result", payload).seq)
            second = request(turn, 2)
            if consumed is not None:
                saved = writer.append("assistant_message", AssistantMessagePayload(
                    turn_id=turn, request_id=second, content="Evidence recorded.", finish_reason="stop",
                    consumed_call_ids=tuple(call.call_id for call in consumed),
                ))
                writer.append("turn_finished", TurnFinishedPayload(
                    turn_id=turn, status="completed", reason="model_stop", assistant_seq=saved.seq,
                ))
            else:
                writer.append("turn_finished", TurnFinishedPayload(turn_id=turn, status="failed", reason="timeout"))
            return result_seqs

        old_user_seq = len(writer.history) + 1
        old_seqs = turn_with_batch(old_calls, task=OLD, consumed=old_calls)
        turn_with_batch((unread,), task="This closed turn has no subsequent assistant reply.")
        turn_with_batch((unknown,), task="Completion is unknown.", consumed=(unknown,), is_unknown=True)
        turn_with_batch((recent,), task="This is the most recent batch.")

    configured = settings(thinking=True)
    monkeypatch.setattr(Settings, "from_env", lambda _env: configured)
    monkeypatch.setattr(cli, "_run_environment_prompt", lambda *_args: lambda: "Fixed runtime snapshot.")

    def layers_through_l2(builder, estimator, policy, *, encode, enable_checkpoint, **_summary_options):
        assert enable_checkpoint
        return manager.ContextWindowManager(builder, estimator, policy, encode=encode)

    monkeypatch.setattr(agent, "ContextWindowManager", layers_through_l2)
    log = store.root / "sessions" / f"{session_id}.jsonl"
    args = ["resume", str(session_id), "--data-root", str(store.root), "--message", LATEST]
    return store, session_id, log, args, (old_user_seq, *old_seqs), (*old_calls, unread, unknown, recent)


def _wire_tokens(payload):
    body = deepcopy(payload)
    body.update(body.pop("extra_body", {}))
    body_size = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
    return (body_size + 2) // 3 + 3 * (len(body["messages"]) + 1)


def test_l2_layers_reach_fake_sdk_and_reuse_view_on_transient_retry(history_session, monkeypatch, capsys):
    store, session_id, log, args, targets, calls = history_session
    before = log.read_bytes()
    prior = SessionReplay(store).replay(session_id, "Replay")
    assert prior.consumed_call_ids == {call.call_id for call in (*calls[:2], calls[3])}
    command_result, search_result = (prior.state.history[seq - 1].payload for seq in targets[1:])
    starts = []

    class SDK(FakeSDK):
        def create(self, **kwargs):
            events = [json.loads(line) for line in log.read_bytes().splitlines()]
            assert events[-1]["type"] == "model_request_started"
            commits = [e for e in events if e["type"] == "compaction_committed"]
            assert len(commits) == 1 and commits[0]["seq"] < events[-1]["seq"]
            starts.append(events[-1]["payload"])
            if len(starts) == 2:
                assert [e["type"] for e in events[len(prior.state.history):]] == [
                    "user_message", "compaction_committed", "model_request_started", "model_request_started",
                ]
            return super().create(**kwargs)

    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    sdk = SDK(timeout, sdk_reply(calls=(sdk_call(),), finish="tool_calls"), sdk_reply("done"))
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    assert cli.main([*args, "--compact-context", "3650", "150", "--measure-context"]) == 0
    sent, retried, followup = sdk.requests
    assert sent == retried
    assert [item["trigger"] for item in starts] == ["initial", "retry_transient", "after_tools"]
    assert len({item["request_id"] for item in starts}) == 3
    assert sent["model"] == "deepseek4.1f" and sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["tools"] == followup["tools"]
    assert sent["messages"][-1] == {"role": "user", "content": LATEST}
    assert sent["messages"][0] == followup["messages"][0]
    assert "Keep failure evidence." in sent["messages"][0]["content"]
    assert "Fixed runtime snapshot." in sent["messages"][0]["content"]
    assert "value = 42" in followup["messages"][-1]["content"]

    results = [message for message in sent["messages"] if message["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == [call.wire_id for call in calls]
    assert [item["id"] for m in sent["messages"] for item in m.get("tool_calls", [])] == [
        call.wire_id for call in calls
    ]
    assert HEAD in results[0]["content"] and TAIL in results[0]["content"] and FAILURE in results[0]["content"]
    assert "RuntimeError" in results[0]["content"]
    assert "pkg/totals.py:" in results[1]["content"] and "pkg/summary.py:" in results[1]["content"]
    original_tools = [message.content for message in prior.projection.messages if message.role == "tool"]
    assert [result["content"] for result in results[2:]] == original_tools[2:]

    logs = [json.loads(s.removeprefix("[context measurement] ")) for s in capsys.readouterr().err.splitlines()
            if s.startswith("[context measurement] ")]
    assert len(logs) == 3 and logs[0]["compaction"] == logs[1]["compaction"]
    layers = logs[0]["compaction"]["layers"]
    assert [item["level"] for item in layers] == ["L1", "L2"] and all(item["selected"] for item in layers)
    assert tuple(layers[0]["target_seqs"]) == targets[:1] and tuple(layers[1]["target_seqs"]) == targets[1:]
    original = deepcopy(sent)
    sent_text = OLD.strip("\n") + "\n"
    next(m for m in original["messages"] if m["content"] == sent_text)["content"] = OLD
    original_results = [message for message in original["messages"] if message["role"] == "tool"]
    original_results[0]["content"] = command_result.output
    original_results[1]["content"] = search_result.output
    assert _wire_tokens(original) == layers[0]["original_tokens"]
    after_l1 = deepcopy(original)
    next(m for m in after_l1["messages"] if m["content"] == OLD)["content"] = sent_text
    assert _wire_tokens(after_l1) == layers[0]["candidate_tokens"] == layers[1]["original_tokens"]
    assert _wire_tokens(sent) == logs[0]["estimate"]["estimated_input_tokens"] == layers[1]["candidate_tokens"]

    after = SessionReplay(JsonlEventStore(store.root)).replay(session_id, "Replay")
    assert log.read_bytes().startswith(before) and after.state.history[:len(prior.state.history)] == prior.state.history
    commit, = [e.payload for e in after.state.history if e.type == "compaction_committed"]
    assert commit.levels == ("L1", "L2")
    by_seq = {item.target_seq: item.content for item in commit.replacements}
    assert set(by_seq) == set(targets)
    assert [by_seq[seq] for seq in targets[1:]] == [m["content"] for m in results[:2]]
    fresh_replies = [e.payload for e in after.state.history[len(prior.state.history):] if e.type == "assistant_message"]
    assert fresh_replies[0].consumed_call_ids == tuple(call.call_id for call in calls)
    assert fresh_replies[1].consumed_call_ids == (*fresh_replies[0].consumed_call_ids,
                                                 fresh_replies[0].tool_calls[0].call_id)
    assert after.consumed_call_ids == set(fresh_replies[1].consumed_call_ids)
    before_restart = log.read_bytes()
    child, (observed,) = _run_cli(
        log.parent.parent, [*args[:-1], "Resume after restarting.", "--python", sys.executable],
        journal=log, trace=log.parent.parent / "child-trace.jsonl", terminal=False,
        replies=["Resumed with the saved evidence."],
    )
    assert "[completed] model_stop" in child.stdout
    assert observed["request"]["messages"][1:-1] == [
        message.model_dump(mode="json") for message in after.projection.messages[1:]
    ]
    assert log.read_bytes().startswith(before_restart)
    assert [e.type for e in store.iter_events(session_id)].count("compaction_committed") == 1
    print(json.dumps({
        "levels": [dict(layer) for layer in layers], "selected": logs[0]["estimate"]["estimated_input_tokens"],
        "command_excerpt": results[0]["content"], "search_excerpt": results[1]["content"],
        "tool_ids": [m["tool_call_id"] for m in results],
    }))


@pytest.mark.parametrize("failure", ["extraction_failed", "invalid_candidate", "no_gain"])
def test_l2_rejected_candidate_sends_only_the_legal_l1_view(history_session, monkeypatch, capsys, failure):
    store, session_id, log, args, targets, _ = history_session
    prior = SessionReplay(store).replay(session_id, "Replay")
    before = log.read_bytes()
    sdk = FakeSDK(sdk_reply("done"))
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    if failure == "extraction_failed":
        def fail(_history):
            raise RuntimeError("simulated extractor failure")

        monkeypatch.setattr(manager, "l2_replacements", fail)
    elif failure == "invalid_candidate":
        monkeypatch.setattr(manager, "l2_replacements", lambda _history: (
            TextReplacement(target_seq=targets[1], content="Not a verified excerpt"),
        ))
    else:
        # Force an equal wire estimate after an otherwise valid L2 preview.
        estimates = iter((6579, 4846, 4846))
        monkeypatch.setattr(manager.ContextWindowManager, "_tokens", lambda *_args: next(estimates))

    assert cli.main([*args, "--compact-context", "5750", "150", "--measure-context"]) == 0
    request, = sdk.requests
    line, = [s for s in capsys.readouterr().err.splitlines() if s.startswith("[context measurement] ")]
    comparison = json.loads(line.removeprefix("[context measurement] "))["compaction"]
    first, second = comparison["layers"]
    assert first["selected"] and second["triggered"] and not second["selected"]
    assert second["status"] == failure
    assert [m["content"] for m in request["messages"] if m["role"] == "tool"] == [
        message.content for message in prior.projection.messages if message.role == "tool"
    ]
    assert any(m["content"] == OLD.strip("\n") + "\n" for m in request["messages"])
    assert log.read_bytes().startswith(before)
    committed, = [e.payload for e in store.iter_events(session_id) if e.type == "compaction_committed"]
    assert committed.levels == ("L1",)
    assert [item.target_seq for item in committed.replacements] == [targets[0]]
