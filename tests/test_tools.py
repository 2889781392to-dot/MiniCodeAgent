import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import BaseModel

from minicodeagent.config import API_KEY_ENV
from minicodeagent.environment import isolated_environment
from minicodeagent.protocol import ToolCall, ToolResult
from minicodeagent import workspace as workspace_module
from minicodeagent.tools import builtin, process
from minicodeagent.tools.builtin import MAX_FILE_BYTES, register_builtin_tools
from minicodeagent.tools.registry import ToolDefinition, ToolRegistry
from minicodeagent.workspace import Workspace, WorkspaceError


def invoke(registry, workspace, name, arguments):
    return registry.execute(
        ToolCall(provider_call_id="from-model", name=name, arguments_raw=json.dumps(arguments)), workspace,
    )


class NumberArgs(BaseModel):
    number: int


def test_registry_binding_and_validation(tmp_path):
    registry = ToolRegistry()
    workspace = Workspace(tmp_path)

    def handler(args, workspace, call_id):
        return ToolResult(call_id=call_id, status="ok", output=str(args.number))

    definition = ToolDefinition("number", "Show number", NumberArgs, handler)
    registry.register(definition)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition)
    with registry.fixed_bindings(), pytest.raises(RuntimeError, match="in use"):
        registry.register(ToolDefinition("other", "", NumberArgs, handler))

    registry.definitions()[0].parameters.clear()
    assert registry.definitions()[0].parameters["required"] == ["number"]
    assert invoke(registry, workspace, "number", {"number": 2}).output == "2"
    assert invoke(registry, workspace, "missing", {}).details["error_code"] == "unknown_tool"
    for raw in ("[]", "invalid", "{}", '{"number":"2"}', '{"number":2,"extra":0}', '{"number":NaN}'):
        call = ToolCall(provider_call_id="id", name="number", arguments_raw=raw)
        error = registry.execute(call, workspace)
        assert error.details["error_code"] == "invalid_arguments"
        assert error.call_id == call.call_id


@pytest.mark.parametrize("behavior", ["exception", "none", "wrong_id", "not_json", "oversized_details"])
def test_bad_handler_result_is_settled(tmp_path, behavior):
    def handler(args, workspace, call_id):
        if behavior == "exception":
            raise RuntimeError("private diagnostic")
        if behavior == "none":
            return None
        if behavior == "wrong_id":
            return ToolResult(call_id=uuid4(), status="ok", output="")
        details = {"data": object() if behavior == "not_json" else "x" * (140 * 1024)}
        return ToolResult.model_construct(call_id=call_id, status="ok", output="", details=details)

    registry = ToolRegistry()
    registry.register(ToolDefinition("number", "", NumberArgs, handler))
    call = ToolCall(provider_call_id="id", name="number", arguments_raw='{"number":1}')
    result = registry.execute(call, Workspace(tmp_path))
    assert result.status == "error"
    assert result.call_id == call.call_id
    assert result.details["error_code"] == "tool_error"
    assert "private diagnostic" not in result.output


def test_workspace_rejects_siblings_and_absolute_escape(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    sibling = tmp_path / "work-other"
    sibling.mkdir()
    workspace = Workspace(root)
    for name in ("../work-other/a.py", str(sibling / "a.py")):
        with pytest.raises(WorkspaceError, match="outside"):
            workspace.resolve_path(name, expect="write")
    assert workspace.resolve_path(str(root / "a.py"), expect="write") == root.resolve() / "a.py"


def check_escape_link(root, outside):
    workspace = Workspace(root)
    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=lambda argv, cwd: False)
    assert invoke(registry, workspace, "write_file", {"path":"external/new.py","content":"bad"}).status == "error"
    assert not (outside / "new.py").exists()
    assert "external" not in invoke(registry, workspace, "list_files", {}).output
    assert "outside sentinel" not in invoke(registry, workspace, "search_code", {"query":"outside sentinel"}).output


def test_outside_link_is_not_walked_or_written(tmp_path):
    root, outside = tmp_path / "work", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.py").write_text("outside sentinel", encoding="utf-8")
    try:
        (root / "external").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip("Windows account cannot create a directory symlink (WinError 1314).")
        raise
    check_escape_link(root, outside)


@pytest.mark.skipif(os.name != "nt", reason="Directory junctions are a Windows feature.")
def test_outside_junction_is_not_walked_or_written(tmp_path):
    root, outside = tmp_path / "work", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.py").write_text("outside sentinel", encoding="utf-8")
    link = root / "external"
    junction_denied = "MCA_JUNCTION_WINERROR_5"
    command = (
        "$ErrorActionPreference = 'Stop'; "
        "try { New-Item -ItemType Junction -Path $env:MCA_TEST_LINK -Target $env:MCA_TEST_TARGET | Out-Null } "
        "catch [System.ComponentModel.Win32Exception] { "
        "if ($_.Exception.NativeErrorCode -eq 5) { "
        f"Write-Output '{junction_denied}'; exit 5 "
        "}; throw }"
    )
    environment = isolated_environment(os.environ, omit_key=API_KEY_ENV)
    environment.update(MCA_TEST_LINK=str(link), MCA_TEST_TARGET=str(outside))
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         command],
        cwd=tmp_path, capture_output=True, text=True, errors="replace", env=environment,
    )
    if result.returncode == 5 and result.stdout.strip() == junction_denied:
        pytest.skip("Windows policy denied directory junction creation (WinError 5).")
    assert result.returncode == 0, result.stderr
    try:
        check_escape_link(root, outside)
    finally:
        link.rmdir()


def test_walk_files_skips_entries_marked_as_reparse_points(tmp_path, monkeypatch):
    from contextlib import contextmanager

    root = tmp_path.resolve()
    folder = root / "reparse_fixture"
    folder.mkdir()
    (folder / "hidden.py").write_text("hidden", encoding="utf-8")
    visible = root / "visible.py"
    visible.write_text("visible", encoding="utf-8")
    scan = os.scandir

    @contextmanager
    def scan_with_reparse_metadata(path):
        assert path != folder, "traversed a directory marked as a reparse point"
        with scan(path) as entries:
            yield (
                SimpleNamespace(
                    path=entry.path,
                    stat=lambda *, follow_symlinks: SimpleNamespace(
                        st_mode=stat.S_IFDIR,
                        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                    ),
                ) if entry.name == folder.name else entry
                for entry in entries
            )

    monkeypatch.setattr(workspace_module, "os", SimpleNamespace(name=os.name, scandir=scan_with_reparse_metadata))
    workspace = Workspace(root)
    assert list(workspace.walk_files(root)) == [visible]


@pytest.mark.parametrize("file_kind", ["missing", "directory", "invalid_utf8", "too_large", "unreadable"])
def test_read_file_returns_paired_tool_error_for_bad_file(tmp_path, monkeypatch, file_kind):
    workspace = Workspace(tmp_path)
    target = workspace.root / "input"
    if file_kind == "directory":
        target.mkdir()
    elif file_kind == "invalid_utf8":
        target.write_bytes(b"\xff")
    elif file_kind == "too_large":
        target.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    elif file_kind == "unreadable":
        target.write_bytes(b"sample")
        original_open = Path.open

        def deny_fixture_read(path, *args, **kwargs):
            if path == target:
                raise PermissionError("fixture access denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", deny_fixture_read)

    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=lambda argv, cwd: False)
    call = ToolCall(provider_call_id="read", name="read_file", arguments_raw='{"path":"input"}')
    result = registry.execute(call, workspace)
    assert result.call_id == call.call_id
    assert result.status == "error"
    assert result.details["error_code"] == "tool_error"
    assert result.output
    assert "fixture access denied" not in result.output


def test_file_tools_share_scan_rules_and_exact_edit(tmp_path):
    workspace = Workspace(tmp_path)
    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=lambda argv, cwd: False)
    assert {spec.name for spec in registry.definitions()} == {
        "list_files", "read_file", "search_code", "write_file", "edit_file", "run_command",
    }
    assert invoke(registry, workspace, "write_file", {"path":"src/a.py","content":"hello\nhello\n"}).status == "ok"
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored").write_text("hello", encoding="utf-8")
    (tmp_path / "binary").write_bytes(b"\0hello")
    assert invoke(registry, workspace, "list_files", {}).output.splitlines() == ["binary", "src/a.py"]
    result = invoke(registry, workspace, "read_file", {"path":"src/a.py","line_count":1})
    assert result.details["next_start_line"] == 2
    assert result.output.startswith("1: hello")
    assert invoke(registry, workspace, "read_file", {"path":"src/a.py","line_count":201}).details["error_code"] == "invalid_arguments"
    assert invoke(registry, workspace, "search_code", {"query":"hello"}).output.splitlines() == [
        "src/a.py:1: hello", "src/a.py:2: hello",
    ]
    for old_text in ("missing", "hello"):
        result = invoke(registry, workspace, "edit_file", {"path":"src/a.py","old_text":old_text,"new_text":"new"})
        assert result.status == "error"
    assert (tmp_path / "src/a.py").read_bytes() == b"hello\nhello\n"
    assert invoke(registry, workspace, "edit_file", {"path":"src/a.py","old_text":"hello\nhello","new_text":"new"}).status == "ok"
    assert (tmp_path / "src/a.py").read_bytes() == b"new\n"


@pytest.mark.parametrize("suffix", [b"", b"partial line"], ids=["exact_budget", "over_budget"])
def test_search_code_handles_file_at_scan_budget(tmp_path, monkeypatch, suffix):
    query = "你好"
    complete_line = f"{query}\n".encode("utf-8")
    (tmp_path / "sample.txt").write_bytes(complete_line + suffix)
    monkeypatch.setattr(builtin, "MAX_SEARCH_BYTES", len(complete_line))
    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=lambda argv, cwd: False)

    result = invoke(registry, Workspace(tmp_path), "search_code", {"query": query})

    assert result.status == "ok"
    assert result.details["scan_byte_limit_reached"] is True
    if suffix:
        assert result.details["count"] == 0
        assert query not in result.output
    else:
        assert result.details["count"] == 1
        assert result.output == f"sample.txt:1: {query}"


@pytest.mark.parametrize("denial", ["no", "eof", "interrupt"])
def test_confirmation_never_starts_rejected_command(tmp_path, monkeypatch, denial):
    def confirm(argv, cwd):
        assert argv == ("python", "-V")
        assert cwd == tmp_path.resolve()
        if denial == "eof":
            raise EOFError
        if denial == "interrupt":
            raise KeyboardInterrupt
        return False

    def unexpected_popen(*args, **kwargs):
        pytest.fail("process started before approval")

    monkeypatch.setattr(process.subprocess, "Popen", unexpected_popen)
    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=confirm)
    result = invoke(registry, Workspace(tmp_path), "run_command", {"argv":["python", "-V"]})
    assert result.details["execution_outcome"] == "not_started"
    assert result.status == ("interrupted" if denial == "interrupt" else "error")


def test_command_uses_approved_args_and_captures_both_streams(tmp_path, monkeypatch):
    approved = []
    registry = ToolRegistry()
    def confirm(argv, cwd):
        approved.append((argv, cwd))
        return True

    register_builtin_tools(registry, confirm_command=confirm)
    monkeypatch.setenv("MINICODEAGENT_API_KEY", "test-secret-never-inherited")
    argv = [
        sys.executable, "-c",
        "import os,sys; print(sys.argv[1]); print(os.getcwd(), file=sys.stderr); "
        "assert 'MINICODEAGENT_API_KEY' not in os.environ; sys.exit(4)",
        "one literal argument & more",
    ]
    result = invoke(registry, Workspace(tmp_path), "run_command", {"argv":argv})
    assert approved == [(tuple(argv), tmp_path.resolve())]
    assert result.details["exit_code"] == 4
    assert result.status == "error"
    assert result.details["stdout"].strip() == argv[-1]
    assert Path(result.details["stderr"].strip()).resolve() == tmp_path.resolve()


def test_real_command_excludes_openai_variables_and_project_key(tmp_path, monkeypatch):
    class SyntheticEnvironment(dict):
        def __getitem__(self, name):
            key = name.upper()
            assert not key.startswith("OPENAI_") and key != API_KEY_ENV
            return super().__getitem__(name)

    environment = SyntheticEnvironment(isolated_environment(os.environ, omit_key=API_KEY_ENV))
    environment.update({
        "OPENAI_API_KEY": "fixture",
        "openai_custom_headers": "fixture",
        "MINICODEAGENT_API_KEY": "fixture",
        "MCA_TEST_PASSTHROUGH": "ordinary setting",
    })
    monkeypatch.setattr(process, "os", SimpleNamespace(name=os.name, environ=environment))

    result = process.execute_command(
        (sys.executable, "-B", "-c",
         "import os; assert not any(k.upper().startswith('OPENAI_') or "
         "k.upper() == 'MINICODEAGENT_API_KEY' for k in os.environ); "
         "print(os.environ['MCA_TEST_PASSTHROUGH'])"),
        tmp_path.resolve(), 5, uuid4(), lambda argv, cwd: True,
    )
    assert result.status == "ok"
    assert result.details["exit_code"] == 0
    assert result.details["stdout"].strip() == "ordinary setting"


def test_approved_command_does_not_start_if_cwd_disappears(tmp_path, monkeypatch):
    cwd = tmp_path / "original"
    cwd.mkdir()

    def confirm(argv, approved_cwd):
        assert approved_cwd == cwd
        cwd.rename(tmp_path / "renamed")
        return True

    def unexpected_popen(*args, **kwargs):
        pytest.fail("started a command in a changed directory")

    monkeypatch.setattr(process.subprocess, "Popen", unexpected_popen)
    result = process.execute_command(("python", "-V"), cwd, 5, uuid4(), confirm)
    assert result.status == "error"
    assert result.details["execution_outcome"] == "not_started"


def test_timeout_and_bounded_stdout_and_stderr(tmp_path):
    registry = ToolRegistry()
    workspace = Workspace(tmp_path)
    register_builtin_tools(registry, confirm_command=lambda argv, cwd: True)
    result = invoke(registry, workspace, "run_command", {
        "argv":[sys.executable,"-u","-c","import time; print('before timeout'); time.sleep(8)"],
        "timeout_seconds":0.5,
    })
    assert result.status == "timeout"
    assert "before timeout" in result.output
    result = invoke(registry, workspace, "run_command", {
        "argv":[sys.executable,"-c","import sys; print('a'*50000); print('b'*50000, file=sys.stderr)"],
    })
    assert result.status == "ok"
    for field in ("stdout", "stderr"):
        assert result.details[f"{field}_truncated"]
        assert len(result.details[field]) < process.MAX_STREAM_BYTES + 100


def test_interrupt_stops_direct_child(tmp_path, monkeypatch):
    original_wait = subprocess.Popen.wait
    calls = 0
    running = []

    def interrupted_wait(self, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            running.append(self)
            raise KeyboardInterrupt
        return original_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupted_wait)
    result = process.execute_command(
        (sys.executable, "-c", "import time; time.sleep(8)"),
        tmp_path, 5, uuid4(), lambda argv, cwd: True,
    )
    assert result.status == "interrupted"
    assert result.details["execution_outcome"] == "unknown"
    assert running[0].poll() is not None
