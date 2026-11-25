"""L4 acceptance with real estimators, bounded fake replies and validated history."""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from minicodeagent.context.budget import BudgetPolicy, SystemPromptParts, TokenEstimator
from minicodeagent.context.builder import ContextBuilder, l3_bounds, l3_checkpoint, l4_source
from minicodeagent.context.events import CompactionCommittedPayload, ExtractiveCheckpoint
from minicodeagent.context.compression.l4 import CodingHandoff, SummaryBudget
from minicodeagent.context.manager import ContextWindowManager
from minicodeagent.context.writer import EventWriteError, MemoryEventSink, SessionEventWriter
from minicodeagent.context.replay import SessionReducer
from minicodeagent.protocol import ChatRequest, ModelError, ModelReply, TokenUsage


GOAL = "Continue the latest task. Do not change tests."
OLD_TODO = "TODO: recheck the fixture."
NOISE = "An obsolete discussion line carried no executable facts.\n" * 145
PARTS = SystemPromptParts("Coding agent.", "\nproject rules", "\nruntime facts")


@pytest.fixture
def history():
    writer = SessionEventWriter()
    writer.append("session_created", {"title": "Fixture", "workspace": "/fixed/project"})
    for i in range(3):
        turn = uuid4()
        writer.append("user_message", {"turn_id": turn, "content": f"Old goal {i}.\n" + NOISE})
        writer.append("assistant_message", {
            "turn_id": turn, "request_id": uuid4(), "content": f"Report {i}.\n{OLD_TODO}\n" + NOISE,
            "finish_reason": "stop",
        })
        writer.append("turn_finished", {
            "turn_id": turn, "status": "completed", "reason": "model_stop", "assistant_seq": writer.history[-1].seq,
        })
    writer.append("user_message", {"turn_id": uuid4(), "content": GOAL})
    return writer.history


def setup(history, fake, budget=SummaryBudget()):
    builder = ContextBuilder()
    estimator = TokenEstimator()
    original = builder.project(history, PARTS.text)
    cp = l3_checkpoint(history)
    compacted = builder.project(history, PARTS.text, levels=("L3",), checkpoint=cp)
    l3_tokens = estimator.estimate(ChatRequest(messages=compacted.messages)).estimated_input_tokens
    # Put L3 above the target and the hard cap above L3. Use an actual estimate.
    policy = BudgetPolicy((l3_tokens * 125 + 94) // 95, 0)
    manager = ContextWindowManager(builder, estimator, policy, enable_checkpoint=True,
                                   summarizer=fake, summary_budget=budget)
    return manager, original


def make_handoff(request, mutate=None):
    data = json.loads(request.messages[1].content)
    result = {"task_goal": data["task_goal"], "evidence": [data["evidence"][0]["id"]]}
    if mutate is not None:
        mutate(result, data)
    return json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("verbatim", [False, True], ids=["reference", "verbatim"])
def test_handoff_preview_selects_grounded_summary_and_preserves_tail(history, verbatim):
    requests = []

    def fake(request, *, max_output_tokens):
        assert not request.tools and max_output_tokens == 1200
        assert [message.role for message in request.messages] == ["system", "user"]
        assert "CodingHandoff" in request.messages[0].content
        assert "project rules" not in request.messages[1].content and NOISE not in request.messages[1].content
        requests.append(request)

        def select_literal(handoff, source):
            if verbatim:
                handoff["evidence"] = [source["evidence"][0]["citation"]]

        return ModelReply(content=make_handoff(request, select_literal), finish_reason="stop")

    manager, original = setup(history, fake)
    result = manager.prepare(history, original, PARTS, ())
    assert len(requests) == 1
    assert result.compaction.levels == ("L4",)
    assert result.compaction.checkpoint.kind == "handoff"
    assert result.compaction.checkpoint.covered_through_seq == l3_bounds(history).covered_through_seq
    source = json.loads(requests[0].messages[1].content)
    assert result.compaction.checkpoint.summary.evidence == (source["evidence"][0]["citation"],)
    assert source["required_references"] == []
    assert result.diagnostic.summary.input_tokens > 0
    assert result.diagnostic.summary.handoff_tokens > 0
    assert result.request.messages[0] == original.messages[0]
    assert result.request.messages[-1] == original.messages[-1]
    assert [m.content.startswith("[Coding handoff") for m in result.request.messages].count(True) == 1
    assert SessionReducer().replay(history).checkpoint is None


@pytest.mark.parametrize("bad", [
    "extra", "type", "duplicate_key", "goal", "seq", "call", "unverified_completed", "oversized",
    "unknown_reference", "wrong_citation_field", "duplicated_reference", "paraphrased_citation",
])
def test_handoff_invalid_output_falls_back_and_caches_the_source(history, bad):
    attempts = []

    def fake(request, *, max_output_tokens):
        attempts.append(request)
        if bad == "duplicate_key":
            text = '{"task_goal":"wrong","task_goal":' + json.dumps(GOAL) + '}'
        elif bad == "oversized":
            text = " " * (4 * max_output_tokens)
        else:
            def corrupt(handoff, source):
                if bad == "extra":
                    handoff["new_field"] = False
                elif bad == "type":
                    handoff["open_issues"] = "not a list"
                elif bad == "goal":
                    handoff["task_goal"] = "Replace the tests."
                elif bad == "seq":
                    handoff["evidence"] = ["目标 [999]：imaginary"]
                elif bad == "call":
                    handoff["evidence"] = [f"变更 [3]：{uuid4()} edit_file status=ok path=\"x.py\""]
                elif bad == "unverified_completed":
                    handoff["completed_work"] = ["助手报告 [3]：Report 0."]
                elif bad == "unknown_reference":
                    handoff["evidence"] = ["E999"]
                elif bad == "wrong_citation_field":
                    report = next(item for item in source["evidence"] if item["citation"].startswith("助手报告"))
                    assert report["allowed_fields"] == ["evidence"]
                    handoff["completed_work"] = [report["id"]]
                elif bad == "duplicated_reference":
                    first = source["evidence"][0]
                    handoff["evidence"] = [first["id"], first["citation"]]
                elif bad == "paraphrased_citation":
                    handoff["evidence"] = [source["evidence"][0]["citation"] + " (edited)"]
            text = make_handoff(request, corrupt)
        return ModelReply(content=text, finish_reason="stop")

    manager, original = setup(history, fake)
    results = [manager.prepare(history, original, PARTS, ()) for _ in range(2)]
    assert len(attempts) == 1
    assert all(result.compaction.levels == ("L3",) for result in results)
    assert all(result.projection.checkpoint.kind == "extractive" for result in results)


@pytest.mark.parametrize("problem", ["small_input", "estimate_unavailable", "timeout", "provider_usage_limit"])
def test_handoff_budgets_and_provider_failures_keep_deterministic_view(history, monkeypatch, problem):
    attempts = []

    def fake(request, *, max_output_tokens):
        attempts.append(request)
        if problem == "timeout":
            raise ModelError("timeout", "offline")
        return ModelReply(content=make_handoff(request), finish_reason="stop",
                          usage=TokenUsage(input_tokens=10, output_tokens=1201, total_tokens=1211))

    budget = SummaryBudget(input_tokens=2) if problem == "small_input" else SummaryBudget()
    manager, original = setup(history, fake, budget)
    if problem == "estimate_unavailable":
        def unavailable(*args, **kwargs):
            raise ValueError("offline estimator")
        monkeypatch.setattr(manager, "_encode_summary", unavailable)
    result = manager.prepare(history, original, PARTS, ())
    assert len(attempts) == int(problem in {"timeout", "provider_usage_limit"})
    assert result.compaction.levels == ("L3",)
    assert result.diagnostic.summary.output_limit == 1200


@pytest.mark.parametrize("bad", [
    {}, {"task_goal": "   "}, {"task_goal": "g", "unresolved": []},
    {"task_goal": "g", "completed_work": ["v"] * 25}, {"task_goal": "g", "evidence": [5]},
])
def test_coding_handoff_requires_exact_known_schema(bad):
    with pytest.raises(ValidationError):
        CodingHandoff.model_validate_json(json.dumps(bad))


def test_l3_fallback_cannot_discard_citations_from_an_existing_handoff(history):
    def fake(request, *, max_output_tokens):
        source = json.loads(request.messages[-1].content)
        first_report = next(item["id"] for item in source["evidence"] if item["citation"].startswith("助手报告"))
        return ModelReply(content=json.dumps({"task_goal": source["task_goal"], "evidence": [first_report]}),
                          finish_reason="stop")

    manager, original = setup(history, fake)
    prepared = manager.prepare(history, original, PARTS, ())
    assert prepared.compaction.levels == ("L4",)
    sink = MemoryEventSink()
    for event in history:
        sink.append(event)
    writer = SessionEventWriter(session_id=history[0].session_id, sink=sink)
    writer.append("compaction_committed", prepared.compaction)
    writer.append("turn_finished", {"turn_id": history[-1].payload.turn_id, "status": "failed", "reason": "offline"})
    turn = uuid4()
    goal = "Next archived goal."
    constraints = "\n".join(f"Constraint {i}: preserve the associated fixture." for i in range(30))
    source = writer.append("user_message", {"turn_id": turn, "content": f"{goal}\n{constraints}"})
    writer.append("assistant_message", {
        "turn_id": turn, "request_id": uuid4(), "content": "Recorded the constraints.", "finish_reason": "stop",
    })
    writer.append("turn_finished", {
        "turn_id": turn, "status": "completed", "reason": "model_stop", "assistant_seq": writer.history[-1].seq,
    })
    writer.append("user_message", {"turn_id": uuid4(), "content": "Use the latest fixtures."})
    before = writer.history
    bound = l3_bounds(before).covered_through_seq
    assert bound >= source.seq
    summary_request, references = l4_source(before, bound)
    required = json.loads(summary_request.messages[-1].content)["required_references"]
    assert [references[identifier] for identifier in required] == list(prepared.compaction.checkpoint.summary.evidence)
    assert len(references.values()) == len(set(references.values()))
    assert l3_checkpoint(before) is None
    lossy = ExtractiveCheckpoint(
        covered_through_seq=bound, kind="extractive", summary=(f"目标 [{source.seq}]：{goal}",),
    )
    with pytest.raises(EventWriteError):
        writer.append("compaction_committed", CompactionCommittedPayload(
            base_seq=before[-1].seq, levels=("L3",), before_tokens=200, after_tokens=150,
            replacements=(), checkpoint=lossy,
        ))
    assert writer.history == before
    assert SessionReducer().replay(before).checkpoint == prepared.compaction.checkpoint
