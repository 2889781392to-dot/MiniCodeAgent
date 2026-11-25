"""Exercise deterministic L1 against real CLI, disk history, and SDK arguments."""

from copy import deepcopy
import json
from uuid import uuid4

import httpx
import pytest
from openai import APITimeoutError

from minicodeagent import cli
from minicodeagent.chat_payload import encode_chat_payload
from minicodeagent.config import Settings
from minicodeagent.context.budget import TokenEstimator
from minicodeagent.context.compression import l1
from minicodeagent.context.events import (
    AssistantMessagePayload, ModelRequestStartedPayload, ToolResultPayload, TurnFinishedPayload, UserMessagePayload,
)
from minicodeagent.context.manager import ContextWindowManager
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ChatMessage, ChatRequest, ToolCall
from minicodeagent.session.service import SessionService
from minicodeagent.tools.builtin import register_builtin_tools
from minicodeagent.tools.registry import ToolRegistry
from test_llm import FakeSDK, sdk_call, sdk_reply, settings


OLD_TEXT = "\n" * 3_000 + "Keep the old assertion, the failed test, and totals.py:5." + "\n" * 3_000
SHORT_TEXT = "Keep the old assertion, the failed test, and totals.py:5.\n"
TASK = "\n\nInspect the new goal and keep its whitespace.\n\n"


@pytest.fixture
def old_session(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("value = 42\n", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("Preserve project test evidence.", encoding="utf-8")
    store = JsonlEventStore(tmp_path / "data")
    calls = [
        ToolCall(provider_call_id=f"saved-{i}", name="read_file", arguments_raw='{"path":"a.py"}')
        for i in range(3)
    ]
    with SessionService(store).create(workspace) as handle:
        writer = handle.writer
        session_id = handle.session_id

        def start(turn, attempt, trigger):
            req = ModelRequestStartedPayload(turn_id=turn, request_id=uuid4(), attempt=attempt, trigger=trigger)
            writer.append("model_request_started", req)
            return req.request_id

        def begin(prompt, batch):
            turn = uuid4()
            writer.append("user_message", UserMessagePayload(turn_id=turn, content=prompt))
            writer.append("assistant_message", AssistantMessagePayload(
                turn_id=turn, request_id=start(turn, 1, "initial"), content="\n\nTool evidence batch.\n\n",
                finish_reason="tool_calls", tool_calls=batch,
            ))
            return turn

        turn = begin("An older completed goal", tuple(calls[:1]))
        writer.append("tool_result", ToolResultPayload(
            turn_id=turn, call_id=calls[0].call_id, status="ok", output="\n\nOld test result.\n\n",
        ))
        reply = writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn, request_id=start(turn, 2, "after_tools"), content=OLD_TEXT,
            finish_reason="stop", consumed_call_ids=(calls[0].call_id,),
        ))
        writer.append("turn_finished", TurnFinishedPayload(
            turn_id=turn, status="completed", reason="model_stop", assistant_seq=reply.seq,
        ))
        turn = begin("An older goal with unconsumed results", tuple(calls[1:]))
        for call, outcome in zip(calls[1:], ("unknown", "not_started")):
            writer.append("tool_result", ToolResultPayload(
                turn_id=turn, call_id=call.call_id, status="interrupted",
                output=f"\n\nExecution outcome {outcome}.\n\n", details={"execution_outcome": outcome},
            ))
        start(turn, 2, "after_tools")
        writer.append("turn_finished", TurnFinishedPayload(turn_id=turn, status="failed", reason="timeout"))
    configured = settings(thinking=True)
    monkeypatch.setattr(Settings, "from_env", lambda _env: configured)
    monkeypatch.setattr(cli, "_run_environment_prompt", lambda *_args: lambda: "Fixed runtime snapshot.")
    log = store.root / "sessions" / f"{session_id}.jsonl"
    args = ["resume", str(session_id), "--data-root", str(store.root), "--message", TASK]
    return store, session_id, log, args, calls


def test_l1_cli_sends_selected_view_reuses_it_on_retry_and_records_actual_consumption(
    old_session, monkeypatch, capsys,
):
    store, session_id, log, args, old_calls = old_session
    before = log.read_bytes()
    state = SessionReplay(store).replay(session_id, "Replay").state
    assert state.consumed_call_ids == {old_calls[0].call_id}
    recorded = []

    class SDK(FakeSDK):
        def create(self, **kwargs):
            events = [json.loads(line) for line in log.read_bytes().splitlines()]
            assert events[-1]["type"] == "model_request_started"
            commits = [e for e in events if e["type"] == "compaction_committed"]
            assert len(commits) == 1
            assert commits[0]["payload"]["replacements"] == [
                {"target_seq": 7, "content": SHORT_TEXT},
            ]
            recorded.append((events[-1], commits[0]))
            if len(recorded) == 2:
                new_replies = [e for e in events[len(before.splitlines()):] if e["type"] == "assistant_message"]
                assert not new_replies
            return super().create(**kwargs)

    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    sdk = SDK(timeout, sdk_reply(calls=(sdk_call(),), finish="tool_calls"), sdk_reply("done"))
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    assert cli.main([*args, "--compact-context", "3800", "200", "--measure-context"]) == 0
    assert log.read_bytes().startswith(before)
    sent, retry, followup = sdk.requests
    assert sent == retry
    assert [r[0]["payload"]["attempt"] for r in recorded] == [1, 2, 3]
    assert [r[0]["payload"]["trigger"] for r in recorded] == ["initial", "retry_transient", "after_tools"]
    assert len({r[0]["payload"]["request_id"] for r in recorded}) == 3
    assert sent["model"] == "deepseek4.1f" and sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["tools"] == followup["tools"]
    assert SHORT_TEXT in [m["content"] for m in sent["messages"]]
    assert OLD_TEXT not in [m["content"] for m in sent["messages"]]
    assert sent["messages"][-1] == {"role": "user", "content": TASK}
    system = sent["messages"][0]["content"]
    assert system.count("Preserve project test evidence.") == system.count("Fixed runtime snapshot.") == 1
    assert followup["messages"][0] == sent["messages"][0]
    paired = [m for m in sent["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in paired] == [c.wire_id for c in old_calls]
    assert [m["content"] for m in paired] == [
        "\n\nOld test result.\n\n", "\n\nExecution outcome unknown.\n\n", "\n\nExecution outcome not_started.\n\n",
    ]
    assert [a["id"] for m in sent["messages"] for a in m.get("tool_calls", [])] == [c.wire_id for c in old_calls]
    assert "value = 42" in followup["messages"][-1]["content"]
    events = tuple(store.iter_events(session_id))
    saved = [e.payload for e in events[len(before.splitlines()):] if e.type == "assistant_message"]
    assert [str(p.request_id) for p in saved] == [r[0]["payload"]["request_id"] for r in recorded[1:]]
    by_wire_id = {call.wire_id: call.call_id for call in [*old_calls, *saved[0].tool_calls]}
    for payload, actual in zip(saved, (retry, followup)):
        assert payload.consumed_call_ids == tuple(
            by_wire_id[m["tool_call_id"]] for m in actual["messages"] if m["role"] == "tool"
        )

    logs = [json.loads(s.removeprefix("[context measurement] ")) for s in capsys.readouterr().err.splitlines()
            if s.startswith("[context measurement] ")]
    assert len(logs) == 3
    assert [row["request_id"] for row in logs] == [r[0]["payload"]["request_id"] for r in recorded]
    selected = logs[0]["estimate"]["estimated_input_tokens"]
    comparison = logs[0]["compaction"]
    assert comparison == logs[1]["compaction"]
    assert comparison["selected_tokens"] == selected < comparison["original_tokens"]
    assert comparison["status"] in {"target_reached", "reduced_above_target"}
    commit = recorded[0][1]["payload"]
    assert (commit["before_tokens"], commit["after_tokens"]) == (comparison["original_tokens"], selected)
    # Recreate the same canonical wire JSON sizes from what was actually received.
    def measured_wire(payload):
        body = deepcopy(payload)
        body.update(body.pop("extra_body", {}))
        return (len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()) + 2) // 3 + 3 * (
            len(body["messages"]) + 1
        )

    original = deepcopy(sent)
    next(m for m in original["messages"] if m["content"] == SHORT_TEXT)["content"] = OLD_TEXT
    assert (measured_wire(original), measured_wire(sent)) == (comparison["original_tokens"], selected)
    assert log.read_bytes().startswith(before)
    replay = SessionReplay(store).replay(session_id, "Replay")
    assert any(m.content == SHORT_TEXT for m in replay.projection.messages)
    assert any(e.type == "assistant_message" and e.payload.content == OLD_TEXT for e in replay.state.history)
    assert replay.consumed_call_ids == set(saved[-1].consumed_call_ids)
    print(json.dumps({
        "original": comparison["original_tokens"], "selected": selected, "status": comparison["status"],
        "request_ids": [row["request_id"] for row in logs], "replaced_seq": commit["replacements"][0]["target_seq"],
        "old_newline_count": OLD_TEXT.count("\n"), "selected_text": SHORT_TEXT,
        "tool_ids": [p["tool_call_id"] for p in paired], "model": sent["model"],
        "trigger": commit["trigger"], "system": system,
    }))


@pytest.mark.parametrize("flags", [[], ["--compact-context", "100000", "200"]], ids=["disabled", "below-high"])
def test_l1_cli_leaves_actual_request_unchanged_without_trigger(old_session, monkeypatch, flags):
    store, session_id, log, args, _ = old_session
    sdk = FakeSDK(sdk_reply("done"))
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    before = SessionReplay(store).replay(session_id, cli.SYSTEM_PROMPT)
    assert cli.main([*args, *flags]) == 0
    request, = sdk.requests
    system = cli.SYSTEM_PROMPT + (
        "\n\n[Project background from Workspace AGENTS.md]\nPreserve project test evidence.\n"
        "[End project background]\n\nFixed runtime snapshot."
    )
    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=lambda *_args: False)
    expected = ChatRequest(
        messages=(ChatMessage(role="system", content=system), *before.projection.messages[1:],
                  ChatMessage(role="user", content=TASK)),
        tools=registry.definitions(),
    )
    assert request == encode_chat_payload(expected, model="deepseek4.1f", deepseek_thinking="disabled")
    assert all(json.loads(line)["type"] != "compaction_committed" for line in log.read_bytes().splitlines())


@pytest.mark.parametrize("failed_event", ["compaction_committed", "model_request_started"])
def test_commit_or_registration_write_failure_prevents_sdk(old_session, monkeypatch, failed_event):
    store, session_id, log, args, calls = old_session
    before = log.read_bytes()
    original = tuple(store.iter_events(session_id))
    sdk = FakeSDK()
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    # CLI writes through the existing lease, which delegates to _append_locked.
    append_locked = JsonlEventStore._append_locked

    def fail_locked(self, path, event):
        if event.type == failed_event:
            raise OSError("synthetic failure")
        append_locked(self, path, event)

    with monkeypatch.context() as patch:
        patch.setattr(JsonlEventStore, "_append_locked", fail_locked)
        assert cli.main([*args, "--compact-context", "3800", "200"]) == 1
    after = log.read_bytes()
    assert after.startswith(before) and not sdk.requests
    written = [json.loads(line)["type"] for line in after[len(before):].splitlines()]
    if failed_event == "compaction_committed":
        assert written == ["user_message"]
        return
    assert written == ["user_message", "compaction_committed"]

    def forbidden(*_args, **_kwargs):
        pytest.fail("Recovered overlays need no new compaction. Replay also needs no estimator, model, or event write.")

    reopened = JsonlEventStore(store.root)
    with monkeypatch.context() as patch:
        patch.setattr(ContextWindowManager, "prepare", forbidden)
        patch.setattr(l1, "trim_display_edges", forbidden)
        with patch.context() as replay_only:
            replay_only.setattr(TokenEstimator, "estimate", forbidden)
            replay_only.setattr(LLMClient, "complete", forbidden)
            replay_only.setattr(cli, "LLMClient", forbidden)
            replay_only.setattr(SessionEventWriter, "append", forbidden)
            replay = SessionReplay(reopened).replay(session_id, "Replay")
            assert replay.state.history[:len(original)] == original
            assert replay.consumed_call_ids == {calls[0].call_id}
            assert replay.state.current_turn.latest_request is None
            assert any(m.content == SHORT_TEXT for m in replay.projection.messages)
            assert all(m.content != OLD_TEXT for m in replay.projection.messages)
        assert log.read_bytes() == after and not sdk.requests

        resumed_sdk = FakeSDK(sdk_reply("done after reopening"))
        patch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=resumed_sdk))
        next_task = "Resume after the failed registration."
        assert cli.main([*args[:-1], next_task]) == 0

    sent, = resumed_sdk.requests
    recovered_history = ChatRequest(messages=replay.projection.messages[1:])
    assert sent["messages"][1:-1] == encode_chat_payload(recovered_history)["messages"]
    assert sent["messages"][-1] == {"role": "user", "content": next_task}
    final_log = log.read_bytes()
    assert final_log.startswith(after)
    assert [json.loads(line)["type"] for line in final_log[len(after):].splitlines()] == [
        "turn_finished", "user_message", "model_request_started", "assistant_message", "turn_finished",
    ]
    assert tuple(reopened.iter_events(session_id))[:len(original)] == original


def test_measured_hard_limit_prevents_dispatch_if_protected_view_does_not_fit(old_session, monkeypatch, capsys):
    _, _, log, args, _ = old_session
    before = log.read_bytes()
    sdk = FakeSDK()
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    assert cli.main([*args, "--compact-context", "500", "100"]) == 1
    output = capsys.readouterr().out
    assert "context_budget" in output and "candidate_over_capacity" in output
    assert not sdk.requests and log.read_bytes().startswith(before)
    assert [json.loads(line)["type"] for line in log.read_bytes()[len(before):].splitlines()] == [
        "user_message", "turn_finished",
    ]


def test_cli_measurement_observer_failure_does_not_change_task_or_l1(old_session, monkeypatch):
    _, _, log, args, _ = old_session
    sdk = FakeSDK(sdk_reply("done"))
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))

    def broken_display(_snapshot):
        raise OSError("synthetic stderr failure")

    monkeypatch.setattr(cli, "_show_measurement", broken_display)
    assert cli.main([*args, "--compact-context", "3800", "200", "--measure-context"]) == 0
    assert len(sdk.requests) == 1
    assert any(m["content"] == SHORT_TEXT for m in sdk.requests[0]["messages"])
    events = [json.loads(line) for line in log.read_bytes().splitlines()]
    assert events[-1]["payload"]["reason"] == "model_stop"
