"""Verify a provider-window rejection at the actual CLI and SDK boundaries."""

import json

import httpx
import pytest
from openai import APITimeoutError, BadRequestError

from minicodeagent import cli
from minicodeagent.config import LoopLimits, Settings
from minicodeagent.context.budget import TokenEstimator
from minicodeagent.context.events import SessionEvent
from minicodeagent.context.replay import SessionReducer, SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.context.writer import EventWriteError, SessionEventWriter
from minicodeagent.llm import LLMClient
from minicodeagent.protocol import ChatMessage, ChatRequest
from minicodeagent.session.service import SessionService
from minicodeagent.tools.registry import ToolRegistry
from test_l1_cli import OLD_TEXT, SHORT_TEXT, TASK, old_session
from test_l3_cli import archive_session
from test_llm import FakeSDK, sdk_reply, settings


# Both fixtures start below the ordinary high watermark. Only the actual
# provider rejection should trigger compaction on the first model step.
_COMPACT = ["--compact-context", "48000", "120"]


def _rejection():
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.test"))
    return BadRequestError("private provider response", response=response,
                           body={"error": {"code": "context_length_exceeded"}})


def _events(log):
    return [json.loads(line) for line in log.read_bytes().splitlines()]


def _connect(monkeypatch, sdk):
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, sdk_client=sdk))
    monkeypatch.setattr(ToolRegistry, "execute", lambda *_: pytest.fail("History must not reexecute a tool."))


def _input_tokens(wire):
    request = ChatRequest(messages=tuple(ChatMessage(role=m["role"], content=m["content"]) for m in wire["messages"]))
    return TokenEstimator().estimate(request, payload=wire).estimated_input_tokens


@pytest.mark.parametrize("fails_again", [False, True], ids=["recovers", "second-rejection"])
def test_cli_overflow_resends_smaller_payload_once_and_settles_only_valid_reply(old_session, monkeypatch, fails_again):
    store, session_id, log, args, calls = old_session
    original = log.read_bytes()
    baseline = SessionReplay(store).replay(session_id, "Replay").state
    dispatches = []

    class ObservedSDK(FakeSDK):
        def create(self, **kwargs):
            events = _events(log)
            suffix = events[baseline.last_seq:]
            dispatches.append([(event["type"], event["payload"].get("trigger")) for event in suffix])
            state = SessionReducer().replay(SessionEvent.model_validate_json(line) for line in log.read_bytes().splitlines())
            assert state.consumed_call_ids == {calls[0].call_id}
            return super().create(**kwargs)

    sdk = ObservedSDK(_rejection(), _rejection() if fails_again else sdk_reply("Done after window recovery."))
    _connect(monkeypatch, sdk)
    assert cli.main([*args, *_COMPACT, "--measure-context"]) == int(fails_again)
    before, after = sdk.requests
    assert before["tools"] == after["tools"]
    assert before["messages"][0] == after["messages"][0]
    assert before["messages"][-1] == after["messages"][-1] == {"role": "user", "content": TASK}
    assert any(m["content"] == OLD_TEXT for m in before["messages"])
    assert any(m["content"] == SHORT_TEXT for m in after["messages"])
    assert _input_tokens(after) < _input_tokens(before)
    assert [m for m in before["messages"] if m["role"] == "tool"] == [
        m for m in after["messages"] if m["role"] == "tool"
    ]
    assert dispatches == [
        [("user_message", None), ("model_request_started", "initial")],
        [("user_message", None), ("model_request_started", "initial"), ("compaction_committed", "overflow"),
         ("model_request_started", "retry_overflow")],
    ]
    final = SessionReplay(store).replay(session_id, "Replay").state
    assert final.current_turn.model_attempts == 2
    assert final.consumed_call_ids == {call.call_id for call in (calls[:1] if fails_again else calls)}
    assert final.current_turn.finished.reason == ("context_length" if fails_again else "model_stop")
    if fails_again:
        assert final.current_turn.latest_reply is None
    else:
        assert final.current_turn.latest_reply.payload.request_id == final.current_turn.latest_request.request_id
    assert log.read_bytes().startswith(original)


@pytest.mark.parametrize("blocked", ["no-gain", "no-allowance", "disabled"])
def test_cli_overflow_without_legal_view_or_allowance_keeps_first_dispatch(old_session, monkeypatch, blocked):
    store, session_id, log, args, calls = old_session
    consumed = {calls[0].call_id}
    if blocked == "no-gain":
        # With only the current user goal, the full request is protected.
        with SessionService(store).create(store.root.parent / "workspace") as handle:
            session_id = handle.session_id
        log = store.root / "sessions" / f"{session_id}.jsonl"
        args = [args[0], str(session_id), *args[2:]]
        consumed = set()
    elif blocked == "no-allowance":
        configured = settings().model_copy(update={"limits": LoopLimits(max_provider_calls=1)})
        monkeypatch.setattr(Settings, "from_env", lambda _: configured)
    first_seq = len(_events(log))
    sdk = FakeSDK(_rejection())
    _connect(monkeypatch, sdk)
    assert cli.main(args if blocked == "disabled" else [*args, *_COMPACT]) == 1
    suffix = _events(log)[first_seq:]
    assert [event["type"] for event in suffix] == ["user_message", "model_request_started", "turn_finished"]
    assert suffix[-1]["payload"]["reason"] == "context_length"
    assert len(sdk.requests) == 1
    assert SessionReplay(store).replay(session_id, "Replay").state.consumed_call_ids == consumed


@pytest.mark.parametrize("fault", ["compaction_committed", "model_request_started"])
def test_cli_overflow_write_failure_preserves_only_the_initial_sdk_attempt(old_session, monkeypatch, fault):
    store, session_id, log, args, calls = old_session
    first_seq = len(_events(log))
    sdk = FakeSDK(_rejection())
    append = SessionEventWriter.append

    def reject_recovery(writer, event_type, payload):
        if event_type == fault and len(sdk.requests) == 1:
            raise EventWriteError("recovery write failed")
        return append(writer, event_type, payload)

    monkeypatch.setattr(SessionEventWriter, "append", reject_recovery)
    _connect(monkeypatch, sdk)
    assert cli.main([*args, *_COMPACT]) == 1
    suffix = _events(log)[first_seq:]
    expected = ["user_message", "model_request_started"]
    if fault == "model_request_started":
        expected.append("compaction_committed")
    assert [event["type"] for event in suffix] == expected
    assert len(sdk.requests) == 1
    assert SessionReplay(store).replay(session_id, "Replay").state.consumed_call_ids == {calls[0].call_id}


def test_cli_transient_retry_reuses_wire_before_distinct_window_recovery(old_session, monkeypatch):
    _, _, log, args, _ = old_session
    first_seq = len(_events(log))
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    sdk = FakeSDK(timeout, _rejection(), sdk_reply("Recovered."))
    _connect(monkeypatch, sdk)
    assert cli.main([*args, *_COMPACT]) == 0
    initial, transient, recovery = sdk.requests
    assert initial == transient
    assert _input_tokens(recovery) < _input_tokens(transient)
    registered = [e["payload"] for e in _events(log)[first_seq:] if e["type"] == "model_request_started"]
    assert [r["trigger"] for r in registered] == ["initial", "retry_transient", "retry_overflow"]
    assert [r["attempt"] for r in registered] == [1, 2, 3]
    assert len({r["request_id"] for r in registered}) == 3


def test_cli_context_recovery_keeps_only_visible_tool_consumption_when_checkpointing(archive_session, monkeypatch):
    store, session_id, log, args, archived_edit, archived_test, protected = archive_session
    initial = SessionReplay(store).replay(session_id, "Replay").state
    # Arrange a recovery target that requires L3. The normal request starts
    # just below high watermark after accounting for the six built-in schemas.
    baseline = FakeSDK(_rejection(), sdk_reply("Done with the protected result."))
    _connect(monkeypatch, baseline)
    assert cli.main([*args, "--compact-context", "13500", "120"]) == 0
    main = [request for request in baseline.requests if request.get("tools")]
    assert len(main) == 2
    assert len(baseline.requests) == 2
    assert _input_tokens(main[1]) < _input_tokens(main[0])
    before_ids = {m["tool_call_id"] for m in main[0]["messages"] if m["role"] == "tool"}
    after_ids = {m["tool_call_id"] for m in main[1]["messages"] if m["role"] == "tool"}
    assert before_ids != after_ids == {"call_" + protected.payload.call_id.hex}
    final = SessionReplay(store).replay(session_id, "Replay").state
    assert final.current_turn.latest_reply.payload.consumed_call_ids == (protected.payload.call_id,)
    assert final.consumed_call_ids == initial.consumed_call_ids | {protected.payload.call_id}
    assert initial.consumed_call_ids == {archived_edit.payload.call_id, archived_test.payload.call_id}


def test_cli_overflow_timeout_does_not_dispatch_another_recovery(old_session, monkeypatch):
    store, session_id, log, args, calls = old_session
    first_seq = len(_events(log))
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    sdk = FakeSDK(_rejection(), timeout)
    _connect(monkeypatch, sdk)
    assert cli.main([*args, *_COMPACT]) == 1
    assert len(sdk.requests) == 2
    suffix = _events(log)[first_seq:]
    assert [e["payload"]["trigger"] for e in suffix if e["type"] == "model_request_started"] == [
        "initial", "retry_overflow",
    ]
    assert suffix[-1]["payload"]["reason"] == "timeout"
    assert SessionReplay(store).replay(session_id, "Replay").state.consumed_call_ids == {calls[0].call_id}


def test_cli_overflow_recovery_budget_survives_reopening_an_unfinished_turn(old_session, monkeypatch):
    store, session_id, log, args, calls = old_session
    first_seq = len(_events(log))
    first_sdk = FakeSDK(_rejection(), _rejection())
    _connect(monkeypatch, first_sdk)
    append = SessionEventWriter.append

    def fail_terminal(writer, event_type, payload):
        if event_type == "turn_finished":
            raise EventWriteError("terminal write interrupted")
        return append(writer, event_type, payload)

    with monkeypatch.context() as attempt:
        attempt.setattr(SessionEventWriter, "append", fail_terminal)
        assert cli.main([*args, *_COMPACT]) == 1
    assert len(first_sdk.requests) == 2
    before = log.read_bytes()
    state = SessionReplay(JsonlEventStore(store.root)).replay(session_id, "Replay").state
    assert state.current_turn.finished is None and state.current_turn.retried_overflow

    second_sdk = FakeSDK(_rejection())
    _connect(monkeypatch, second_sdk)
    assert cli.main([*args[:-2], "--continue", *_COMPACT]) == 1
    assert second_sdk.requests == [first_sdk.requests[-1]]
    suffix = _events(log)[first_seq:]
    assert [(e["payload"]["attempt"], e["payload"]["trigger"]) for e in suffix
            if e["type"] == "model_request_started"] == [(1, "initial"), (2, "retry_overflow"), (3, "resume")]
    assert sum(e["type"] == "compaction_committed" for e in suffix) == 1
    assert log.read_bytes().startswith(before)
    final = SessionReplay(store).replay(session_id, "Replay").state
    assert final.current_turn.finished.reason == "context_length"
    assert final.consumed_call_ids == {calls[0].call_id}


@pytest.mark.parametrize("commit_fails", [False, True])
def test_cli_overflow_handoff_counts_internal_summary_separately(archive_session, monkeypatch, commit_fails):
    store, session_id, log, args, _, _, protected = archive_session
    initial = SessionReplay(store).replay(session_id, "Replay").state
    summary_calls = []

    class SummarySDK(FakeSDK):
        def create(self, **kwargs):
            if kwargs.get("tools"):
                return super().create(**kwargs)
            summary_calls.append(kwargs)
            suffix = _events(log)[initial.last_seq:]
            assert [e["type"] for e in suffix] == ["user_message", "model_request_started"]
            assert kwargs["max_tokens"] == 1200 and kwargs["tool_choice"] == "none"
            source = json.loads(kwargs["messages"][-1]["content"])
            citation = source["evidence"][0]["id"]
            return sdk_reply(json.dumps({"task_goal": source["task_goal"], "evidence": [citation]}))

    sdk = SummarySDK(_rejection(), sdk_reply("Done."))
    _connect(monkeypatch, sdk)
    append = SessionEventWriter.append

    if commit_fails:
        def fail_commit(writer, event_type, payload):
            if event_type == "compaction_committed":
                raise EventWriteError("recovery checkpoint could not be saved")
            return append(writer, event_type, payload)

        monkeypatch.setattr(SessionEventWriter, "append", fail_commit)

    # Leave a long, protected current goal. L3 reduces the request but
    # remains above the recovery target, making L4 eligible.
    task = args[-1] + "Current goal context must stay protected.\n" * 1_400
    assert cli.main([*args[:-1], task, *_COMPACT]) == int(commit_fails)
    assert len(summary_calls) == 1
    assert len(sdk.requests) == (1 if commit_fails else 2)
    final = SessionReplay(store).replay(session_id, "Replay").state
    if commit_fails:
        assert final.checkpoint is None
        assert final.current_turn.model_attempts == 1
        assert final.consumed_call_ids == initial.consumed_call_ids
    else:
        assert final.checkpoint.kind == "handoff"
        assert final.current_turn.model_attempts == 2
        assert _input_tokens(sdk.requests[1]) < _input_tokens(sdk.requests[0])
        assert final.current_turn.latest_reply.payload.consumed_call_ids == (protected.payload.call_id,)
