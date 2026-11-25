"""Verify L2 excerpts retain observable tool evidence and can be checked read-only."""

import json
from uuid import UUID

import pytest

from minicodeagent.context.compression import l2
from minicodeagent.protocol import ToolCall, ToolResult


def command_result(stdout, stderr="", **extra_details):
    code = extra_details.pop("exit_code", 1)
    call = ToolCall(
        call_id=UUID(int=12), provider_call_id="command", name="run_command",
        arguments_raw=json.dumps({"argv": ["C:/venvs/Project Python/python.exe", "-m", "pytest"], "cwd": "samples"}),
    )
    result = ToolResult(
        call_id=call.call_id, status="error" if code else "ok",
        output=f"Exit code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        details={
            "exit_code": code, "stdout": stdout, "stderr": stderr,
            "stdout_truncated": False, "stderr_truncated": False, "capture_incomplete": False, **extra_details,
        },
    )
    return call, result


@pytest.fixture
def long_failure():
    noise = "running progress tick for the local integration test suite\n" * 55
    stdout = (
        "pytest session begins\ncollecting tests\n" + noise
        + "collected 20 items\n3 errors, 2 warnings\n"
        + "tests/test_totals.py:46: AssertionError\n"
        + "E       assert 3 == 4\n" + noise
        + "FAILED tests/test_totals.py::test_sum\n1 failed, 142 passed in 2.04s\n"
    )
    stderr = (
        "stderr capture start\n" + noise + "Traceback (most recent call last):\n"
        + '  File "src/totals.py", line 8, in sum_values\n'
        + "ZeroDivisionError: division by zero\n" + noise + "stderr capture ends"
    )
    return command_result(stdout, stderr, error_code="command_failed")


def test_run_command_excerpt_preserves_literal_start_end_and_failure_evidence(long_failure):
    call, result = long_failure
    body = l2.extract_tool_result(42, call, result)
    assert body is not None and len(body) < len(result.output)
    assert body == l2.extract_tool_result(42, call, result)
    assert "[L2 tool result excerpt]" in body
    assert '"source_seq":42' in body and str(call.call_id) in body
    assert '"status":"error"' in body and '"exit_code":1' in body
    assert "C:/venvs/Project Python/python.exe" in body and "samples" in body
    assert "pytest session begins" in body and "stderr capture ends" in body
    assert "tests/test_totals.py:46: AssertionError" in body and "assert 3 == 4" in body
    assert "1 failed, 142 passed in 2.04s" in body
    assert "collected 20 items" in body and "3 errors, 2 warnings" in body
    assert "src/totals.py" in body and "line 8" in body and "ZeroDivisionError" in body
    assert '"omitted_lines":' in body
    assert body.count("running progress tick for the local integration test suite") < 20
    assert l2.is_tool_result_excerpt(42, call, result, body)


def test_search_excerpt_preserves_match_locations_one_per_file_and_scan_limits():
    call = ToolCall(
        call_id=UUID(int=14), provider_call_id="search", name="search_code",
        arguments_raw=json.dumps({"path": ".", "query": "needle", "limit": 120}),
    )
    lines = ["src/a.py:1: needle first"]
    lines.extend(f"src/a.py:{line}: needle preceded by a long literal match" for line in range(2, 100))
    lines.append("src/intermediate.py:64: needle from an otherwise omitted file")
    lines.extend(f"tests/test_b.py:{line}: needle in a repeated local match" for line in range(1, 21))
    result = ToolResult(
        call_id=call.call_id, status="ok", output="\n".join(lines),
        details={
            "count": len(lines), "result_limit_reached": True,
            "scan_byte_limit_reached": False, "scan_entry_limit": 900,
        },
    )
    body = l2.extract_tool_result(90, call, result)
    assert body is not None and len(body) < len(result.output)
    for token in ("src/a.py:1: needle first", "src/intermediate.py:64: needle from an otherwise omitted file",
                  "tests/test_b.py:20: needle in a repeated local match", '"result_limit_reached":true',
                  '"scan_byte_limit_reached":false', f'"count":{len(lines)}'):
        assert token in body
    assert l2.is_tool_result_excerpt(90, call, result, body)
    bad_count = result.model_copy(update={"details": {**result.details, "count": 1}})
    assert l2.extract_tool_result(90, call, bad_count) is None


@pytest.mark.parametrize("change", ["unknown", "not_started", "timeout", "wrong_id", "inconsistent",
                                   "incomplete", "registry_truncated", "source", "edit", "small"])
def test_l2_leaves_unproven_or_high_value_output_intact(long_failure, change):
    call, result = long_failure
    if change in {"unknown", "not_started"}:
        result = result.model_copy(update={"details": {**result.details, "execution_outcome": change}})
    elif change == "timeout":
        result = result.model_copy(update={"status": "timeout"})
    elif change == "wrong_id":
        result = result.model_copy(update={"call_id": UUID(int=500)})
    elif change == "inconsistent":
        result = result.model_copy(update={"output": "Exit code: 0\n" + result.output})
    elif change in {"incomplete", "registry_truncated"}:
        key = "capture_incomplete" if change == "incomplete" else "output_truncated"
        result = result.model_copy(update={"details": {**result.details, key: True}})
    elif change in {"source", "edit"}:
        call = call.model_copy(update={"name": "read_file" if change == "source" else "edit_file"})
    else:
        call, result = command_result("ok", exit_code=0)
    assert l2.extract_tool_result(42, call, result) is None
    assert not l2.is_tool_result_excerpt(42, call, result, "fake")


def test_saved_excerpt_is_checked_without_running_extraction(long_failure, monkeypatch):
    call, result = long_failure
    body = l2.extract_tool_result(42, call, result)
    assert body is not None

    def forbidden(*_args, **_kwargs):
        pytest.fail("Replay validation must not generate a new candidate.")

    monkeypatch.setattr(l2, "extract_tool_result", forbidden)
    assert l2.is_tool_result_excerpt(42, call, result, body)
    assert not l2.is_tool_result_excerpt(43, call, result, body)
    assert not l2.is_tool_result_excerpt(42, call, result, body.replace('"exit_code":1', '"exit_code":0'))
    assert not l2.is_tool_result_excerpt(42, call, result, body.replace("assert 3 == 4", "assert 4 == 4"))

    header, payload = body.split("\n", 1)
    excerpt = json.loads(payload)
    stream = excerpt["stdout"]
    stream["lines"] = [pair for pair in stream["lines"] if "AssertionError" not in pair[1]]
    stream["omitted_lines"] += 1
    tampered = header + "\n" + json.dumps(excerpt, separators=(",", ":"), ensure_ascii=False)
    assert not l2.is_tool_result_excerpt(42, call, result, tampered)


def test_rendering_overhead_is_left_to_the_encoded_budget_check():
    call, result = command_result("first\nsecond\nnoise\npenultimate\nlast", exit_code=0)
    body = l2.extract_tool_result(25, call, result)
    assert body is not None and len(body) > len(result.output)
    assert l2.is_tool_result_excerpt(25, call, result, body)
