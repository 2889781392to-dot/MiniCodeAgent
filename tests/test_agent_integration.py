"""Run the real local tools and pytest with only the model scripted."""

import json
import sys
from pathlib import Path
from shutil import copyfile

import pytest

from minicodeagent.agent import AgentLoop
from minicodeagent.config import LoopLimits
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.protocol import ModelError, ModelReply, ToolCall
from minicodeagent.tools.builtin import register_builtin_tools
from minicodeagent.tools.registry import ToolRegistry
from minicodeagent.workspace import Workspace


@pytest.mark.parametrize("retry_after_failure", [False, True], ids=["ordinary", "transient_retry"])
def test_repairs_project_using_real_pytest_feedback(tmp_path, retry_after_failure):
    project = tmp_path / "sample_project"
    project.mkdir()
    source = project / "totals.py"
    acceptance = project / "test_totals.py"
    fixture = Path(__file__).resolve().parents[1] / "examples" / "bugfix"
    copyfile(fixture / source.name, source)
    copyfile(fixture / acceptance.name, acceptance)
    fixed_tests = acceptance.read_bytes()
    argv = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    steps = [
        ("read_file", {"path":"totals.py"}),
        ("edit_file", {"path":"totals.py", "old_text":"return 0", "new_text":"return sum(values)"}),
        ("run_command", {"argv":argv}),
        ("edit_file", {"path":"totals.py", "old_text":"return sum(values)",
                       "new_text":"return sum(value for value in values if value % 2 == 0)"}),
        ("run_command", {"argv":argv}),
    ]

    class ScriptedLLM:
        def __init__(self):
            self.calls = []
            self.requests = []
            self.failed_request = None

        def complete(self, request):
            self.requests.append(request)
            index = len(self.calls)
            if index:
                last_call = self.calls[-1]
                assistant, tool = request.messages[-2:]
                assert assistant.role == "assistant" and assistant.tool_calls == (last_call,)
                assert tool.role == "tool" and tool.call_id == last_call.call_id
                expectations = ["return 0", "Edited totals.py.", "1 failed", "Edited totals.py.", "2 passed"]
                assert expectations[index - 1] in tool.content
                if index == 3:
                    assert "assert 9 == 6" in tool.content
                    if retry_after_failure:
                        if self.failed_request is None:
                            self.failed_request = request
                            raise ModelError("timeout", "Simulated timeout after receiving test feedback.")
                        assert request == self.failed_request
                if index == 5:
                    return ModelReply(content="Fixed totals.py and ran pytest. Both tests passed.", finish_reason="stop")

            name, arguments = steps[index]
            call = ToolCall(provider_call_id=f"model-{index}", name=name, arguments_raw=json.dumps(arguments))
            self.calls.append(call)
            return ModelReply(finish_reason="tool_calls", tool_calls=(call,))

    approvals = []
    def approve(command, cwd):
        approvals.append((command, cwd))
        return True

    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=approve)
    writer = SessionEventWriter()
    model = ScriptedLLM()
    agent = AgentLoop(model, registry, Workspace(project), writer, ContextBuilder(), LoopLimits())
    outcome = agent.run_turn("Fix total_even in totals.py without changing the tests.")
    assert outcome.status == "completed"
    assert source.read_text(encoding="utf-8") == (
        "def total_even(values):\n"
        "    return sum(value for value in values if value % 2 == 0)\n"
    )
    assert acceptance.read_bytes() == fixed_tests
    assert approvals == [(tuple(argv), project.resolve())] * 2
    assert len(model.requests) == 6 + retry_after_failure
    assert [call.name for call in model.calls] == [name for name, _ in steps]
    results = [event.payload for event in writer.history if event.type == "tool_result"]
    assert [result.call_id for result in results] == [call.call_id for call in model.calls]
    assert [result.status for result in results] == ["ok", "ok", "error", "ok", "ok"]
    assert [results[i].details["exit_code"] for i in (2, 4)] == [1, 0]
    assert writer.history[-2].payload.consumed_call_ids == tuple(call.call_id for call in model.calls)
    assert [event.type for event in writer.history] == (
        ["user_message"] + ["assistant_message", "tool_result"] * 5 + ["assistant_message", "turn_finished"]
    )
    assert writer.history[-1].payload.reason == "model_stop"

    for call, result in zip(model.calls, results):
        print(f"{call.name}: {result.status}\n{result.output}")
    print(outcome.content)
