import json

import pytest

from minicodeagent.protocol import ToolCall, ToolResult
from minicodeagent.tools import builtin
from minicodeagent.tools.registry import MAX_DETAILS_BYTES, MAX_OUTPUT_CHARS, ToolRegistry
from minicodeagent.workspace import Workspace


def test_advertised_schema_matches_validation_and_dispatch(tmp_path, monkeypatch):
    calls = []

    def read_file(args, workspace, call_id):
        calls.append((args, workspace, call_id))
        return ToolResult(call_id=call_id, status="ok", output="test fixture")

    monkeypatch.setattr(builtin, "read_file", read_file)
    registry = ToolRegistry()
    builtin.register_builtin_tools(registry, confirm_command=lambda argv, cwd: False)
    workspace = Workspace(tmp_path)
    schema = next(spec.parameters for spec in registry.definitions() if spec.name == "read_file")
    assert schema == builtin.ReadFileArgs.model_json_schema()

    for raw in ('{}', '{"path":"a.py","extra":true}', '{"path":"a.py","start_line":"1"}', '{'):
        call = ToolCall(provider_call_id="invalid", name="read_file", arguments_raw=raw)
        result = registry.execute(call, workspace)
        assert result.status == "error"
        assert result.call_id == call.call_id
        assert result.details["error_code"] == "invalid_arguments"
    call = ToolCall(provider_call_id="unknown", name="not_registered", arguments_raw="{}")
    error = registry.execute(call, workspace)
    assert error.status == "error"
    assert error.call_id == call.call_id
    assert error.details["error_code"] == "unknown_tool"
    assert calls == []

    call = ToolCall(provider_call_id="valid", name="read_file", arguments_raw='{"path":"a.py"}')
    assert registry.execute(call, workspace).status == "ok"
    assert calls == [(builtin.ReadFileArgs(path="a.py"), workspace, call.call_id)]


@pytest.mark.parametrize("details_at_limit", [False, True])
def test_truncated_output_respects_final_details_budget(tmp_path, monkeypatch, details_at_limit):
    details = {"payload": ""}
    if details_at_limit:
        details["payload"] = "x" * (MAX_DETAILS_BYTES - len(json.dumps(details)))
        assert len(json.dumps(details)) == MAX_DETAILS_BYTES

    def read_file(args, workspace, call_id):
        return ToolResult(
            call_id=call_id, status="ok",
            output="begin\n" + "a" * MAX_OUTPUT_CHARS + "\nend",
            details=details,
        )

    monkeypatch.setattr(builtin, "read_file", read_file)
    registry = ToolRegistry()
    builtin.register_builtin_tools(registry, confirm_command=lambda argv, cwd: False)
    call = ToolCall(provider_call_id="large-result", name="read_file", arguments_raw='{"path":"a.py"}')
    result = registry.execute(call, Workspace(tmp_path))
    assert result.call_id == call.call_id
    if details_at_limit:
        assert result.status == "error"
        assert result.details["error_code"] == "tool_error"
        assert "oversized result details" in result.output
    else:
        assert result.status == "ok"
        assert result.details["output_truncated"] is True
        assert result.output.startswith("begin\n")
        assert result.output.endswith("\nend")
    assert len(result.output) <= MAX_OUTPUT_CHARS
    assert len(json.dumps(result.details, allow_nan=False)) <= MAX_DETAILS_BYTES
