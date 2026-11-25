from collections import deque
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from minicodeagent.agent import AgentLoop
from minicodeagent.cli import main
from minicodeagent.config import LoopLimits
from minicodeagent.context.builder import ContextBuilder
from minicodeagent.context.events import ModelRequestStartedPayload, UserMessagePayload
from minicodeagent.context.replay import SessionReplay
from minicodeagent.context.store import JsonlEventStore
from minicodeagent.context.writer import SessionEventWriter
from minicodeagent.project_memory import MAX_PROJECT_MEMORY_BYTES
from minicodeagent.protocol import ModelError, ModelReply, ProtocolModel, ToolCall, ToolResult
from minicodeagent.session.service import SessionService
from minicodeagent.tools.registry import ToolDefinition, ToolRegistry
from minicodeagent.workspace import Workspace


class CaptureModel:
    def __init__(self):
        self.requests = []
        self.replies = deque()

    def complete(self, request):
        self.requests.append(request)
        if self.replies:
            return self.replies.popleft()
        return ModelReply(content="done", finish_reason="stop")


def make_agent(path, client, *, registry=None, writer=None):
    return AgentLoop(
        client, registry if registry is not None else ToolRegistry(), Workspace(path),
        writer if writer is not None else SessionEventWriter(), ContextBuilder(), LoopLimits(),
        system_prompt="System for this test.",
    )


def rules(workspace, text):
    (workspace / "AGENTS.md").write_text(text, encoding="utf-8")


def assert_project_background(request, expected):
    system, = [message for message in request.messages if message.role == "system"]
    assert "project background" in system.content.casefold()
    assert "AGENTS.md" in system.content
    assert system.content.count(expected) == 1
    assert all(expected not in message.content for message in request.messages if message.role != "system")


@pytest.fixture
def cli_model(monkeypatch):
    capture = CaptureModel()

    class Client:
        def __init__(self, _settings):
            pass

        def complete(self, request):
            return capture.complete(request)

        def close(self):
            pass

    monkeypatch.setenv("MINICODEAGENT_API_KEY", "synthetic-project-memory-test-key")
    monkeypatch.setenv("MINICODEAGENT_MODEL", "fake-project-memory-model")
    monkeypatch.setattr("minicodeagent.cli.LLMClient", Client)
    return capture


@pytest.fixture
def cli_requests(cli_model):
    return cli_model.requests


def resume_from_cli(service, session_id, *args):
    return main(["resume", str(session_id), "--data-root", str(service.catalog.root), *args])


def test_missing_root_file_does_not_create_it_or_search_other_directories(tmp_path):
    workspace = tmp_path / "project"
    child = workspace / "subdir"
    child.mkdir(parents=True)
    rules(tmp_path, "Parent project data must stay out.")
    rules(child, "Nested project data must stay out.")
    client = CaptureModel()
    agent = make_agent(workspace, client)

    assert agent.run_turn("first task").status == "completed"
    assert client.requests[0].messages[0].content == "System for this test."
    assert not (workspace / "AGENTS.md").exists()
    assert [event.type for event in agent.writer.history] == [
        "user_message", "assistant_message", "turn_finished",
    ]


def test_cli_two_sessions_share_a_project_and_other_project_isolated(tmp_path, cli_model, capsys):
    shared = tmp_path / "shared"
    other = tmp_path / "other"
    shared.mkdir()
    other.mkdir()
    common_rules = "共享项目规则"
    first_task = "A private task."
    evidence = "Read by Session A."
    (shared / "notes.txt").write_text(evidence, encoding="utf-8")
    rules(shared, common_rules)
    rules(other, "Other project marker.")
    data_root = tmp_path / "data"
    store = JsonlEventStore(data_root)
    service = SessionService(store)

    def create_session(workspace):
        capsys.readouterr()
        assert main(["new", "--workspace", str(workspace), "--data-root", str(data_root)]) == 0
        return UUID(capsys.readouterr().out.strip())

    first = create_session(shared)
    call = ToolCall(provider_call_id="read-a", name="read_file", arguments_raw='{"path":"notes.txt"}')
    cli_model.replies.append(ModelReply(finish_reason="tool_calls", tool_calls=(call,)))
    assert resume_from_cli(service, first, "--message", first_task) == 0
    first_request, after_read = cli_model.requests
    assert_project_background(first_request, common_rules)
    assert_project_background(after_read, common_rules)
    result, = [message for message in after_read.messages if message.role == "tool"]
    assert result.call_id == call.call_id and evidence in result.content

    second = create_session(shared)
    assert resume_from_cli(service, second, "--message", "B private task.") == 0
    second_request = cli_model.requests[-1]
    assert_project_background(second_request, common_rules)
    assert first_request.messages[0].content == second_request.messages[0].content
    assert [message.role for message in second_request.messages] == ["system", "user"]
    assert second_request.messages[1].content == "B private task."

    isolated = create_session(other)
    assert resume_from_cli(service, isolated, "--message", "Different project task.") == 0
    assert_project_background(cli_model.requests[-1], "Other project marker.")
    assert common_rules not in cli_model.requests[-1].messages[0].content
    assert len({first, second, isolated}) == 3
    logs = {session: data_root / "sessions" / f"{session}.jsonl" for session in (first, second, isolated)}
    before = {session: path.read_bytes() for session, path in logs.items()}
    assert evidence.encode() in before[first]
    assert all(marker not in before[second] for marker in (first_task.encode(), evidence.encode(), b"read-a"))
    original_history = tuple(store.iter_events(first))

    new_rules = "Updated by the project maintainer."
    rules(shared, new_rules)
    assert {session: path.read_bytes() for session, path in logs.items()} == before
    assert resume_from_cli(service, second, "--message", "B next task.") == 0
    updated_request = cli_model.requests[-1]
    assert_project_background(updated_request, new_rules)
    assert common_rules not in updated_request.messages[0].content
    history = [message.content for message in updated_request.messages if message.role == "user"]
    assert history == ["B private task.", "B next task."]
    assert not any(message.role == "tool" or message.tool_calls for message in updated_request.messages)
    assert tuple(store.iter_events(first)) == original_history
    assert logs[first].read_bytes() == before[first]
    assert logs[second].read_bytes().startswith(before[second])


def test_cli_override_suspends_memory_and_forks_select_their_bound_projects(
    tmp_path, cli_requests, capsys, monkeypatch,
):
    parent_root = tmp_path / "parent"
    child_root = tmp_path / "branch"
    parent_root.mkdir()
    child_root.mkdir()
    rules(parent_root, "Parent project marker.")
    rules(child_root, "Branch project marker.")
    service = SessionService(JsonlEventStore(tmp_path / "logs"))
    with service.create(parent_root) as parent:
        parent_id = parent.session_id
    with service.fork(parent_id) as shared:
        shared_id = shared.session_id
    with service.fork(parent_id, workspace=child_root) as isolated:
        isolated_id = isolated.session_id

    original_open = Path.open

    def deny_memory_open(path, *args, **kwargs):
        if path.name == "AGENTS.md":
            pytest.fail("A temporary workspace override must skip project memory reads.")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", deny_memory_open)
        assert resume_from_cli(
            service, parent_id, "--workspace", str(child_root), "--message", "temporary task",
        ) == 0
    diagnostic = capsys.readouterr().err.casefold()
    assert "project memory" in diagnostic and "workspace" in diagnostic
    assert "Parent project marker." not in cli_requests[0].messages[0].content
    assert "Branch project marker." not in cli_requests[0].messages[0].content
    with service.open(parent_id) as parent:
        assert parent.state.metadata.workspace == str(parent_root.resolve())

    assert resume_from_cli(service, shared_id, "--message", "shared task") == 0
    assert resume_from_cli(service, isolated_id, "--message", "isolated task") == 0
    assert_project_background(cli_requests[1], "Parent project marker.")
    assert "Branch project marker." not in cli_requests[1].messages[0].content
    assert_project_background(cli_requests[2], "Branch project marker.")
    assert "Parent project marker." not in cli_requests[2].messages[0].content
    assert (parent_root / "AGENTS.md").read_bytes() == b"Parent project marker."
    assert (child_root / "AGENTS.md").read_bytes() == b"Branch project marker."


class NoArgs(ProtocolModel):
    pass


def test_request_retries_and_tool_rounds_reuse_one_snapshot_then_reload(tmp_path, monkeypatch):
    source = tmp_path / "AGENTS.md"
    rules(tmp_path, "Initial project rule.")
    reads = []
    original_open = Path.open

    def count_reads(path, mode="r", *args, **kwargs):
        if path == source and mode.startswith("r"):
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_reads)
    call = ToolCall(provider_call_id="edit-rule", name="probe", arguments_raw="{}")

    class Client(CaptureModel):
        def complete(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                rules(tmp_path, "Edited before retry.")
                raise ModelError("timeout", "retry the request")
            if len(self.requests) == 2:
                return ModelReply(finish_reason="tool_calls", tool_calls=(call,))
            return ModelReply(content="done", finish_reason="stop")

    def probe(_args, workspace, call_id):
        rules(workspace.root, "Edited by tool round.")
        return ToolResult(call_id=call_id, status="ok", output="file updated")

    registry = ToolRegistry()
    registry.register(ToolDefinition("probe", "updates the test project rules", NoArgs, probe))
    client = Client()
    agent = make_agent(tmp_path, client, registry=registry)
    assert agent.run_turn("first task").status == "completed"
    assert len(reads) == 1
    assert len(client.requests) == 3
    for request in client.requests:
        assert_project_background(request, "Initial project rule.")
        assert "Edited" not in request.messages[0].content

    assert agent.run_turn("next task").status == "completed"
    assert len(reads) == 2
    assert_project_background(client.requests[3], "Edited by tool round.")
    assert "Initial project rule." not in client.requests[3].messages[0].content


def test_cli_memory_mode_and_explicit_continue_both_inject_project_background(
    tmp_path, cli_requests,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rules(workspace, "Project marker for existing task.")
    assert main(["--workspace", str(workspace), "--message", "new in memory"]) == 0
    assert_project_background(cli_requests[0], "Project marker for existing task.")

    service = SessionService(JsonlEventStore(tmp_path / "data"))
    with service.create(workspace) as handle:
        session_id = handle.session_id
        turn_id = uuid4()
        handle.writer.append("user_message", UserMessagePayload(turn_id=turn_id, content="saved task"))
        handle.writer.append("model_request_started", ModelRequestStartedPayload(
            turn_id=turn_id, request_id=uuid4(), attempt=1, trigger="initial",
        ))
    rules(workspace, "Project marker after reopening.")
    assert resume_from_cli(service, session_id, "--continue") == 0
    assert_project_background(cli_requests[1], "Project marker after reopening.")
    assert [message.content for message in cli_requests[1].messages if message.role == "user"] == ["saved task"]


def assert_blocked(agent, client, diagnostic):
    before = agent.writer.history
    outcome = agent.run_turn("request with bad rules")
    assert (outcome.status, outcome.reason) == ("failed", "project_memory_error")
    assert "AGENTS.md" in outcome.content
    assert diagnostic in outcome.content.casefold()
    assert client.requests == []
    assert agent.writer.history == before


def test_size_limit_reads_only_the_bounded_prefix(tmp_path, monkeypatch):
    assert MAX_PROJECT_MEMORY_BYTES == 16 * 1024
    source = tmp_path / "AGENTS.md"
    cap = MAX_PROJECT_MEMORY_BYTES
    source.write_bytes(b"x" * (cap * 4))
    original_open = Path.open
    chunks = []

    class InspectRead:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, requested=-1):
            assert 0 < requested <= cap + 1
            data = self.file.read(requested)
            chunks.append(len(data))
            return data

    def inspect_source(path, mode="r", *args, **kwargs):
        file = original_open(path, mode, *args, **kwargs)
        return InspectRead(file) if path == source and mode == "rb" else file

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", inspect_source)
        client = CaptureModel()
        assert_blocked(make_agent(tmp_path, client), client, "limit")
        assert 0 < sum(chunks) <= cap + 1

    source.write_bytes(b"Y" * cap)
    client = CaptureModel()
    assert make_agent(tmp_path, client).run_turn("exact limit").status == "completed"
    assert_project_background(client.requests[0], "Y" * cap)


def test_invalid_utf8_blocks_request_before_persisting_turn(tmp_path):
    (tmp_path / "AGENTS.md").write_bytes(b"Invalid \xff")
    client = CaptureModel()
    assert_blocked(make_agent(tmp_path, client), client, "utf-8")


def test_existing_unreadable_file_blocks_request(tmp_path, monkeypatch):
    rules(tmp_path, "This exists.")
    original_open = Path.open
    source = tmp_path / "AGENTS.md"

    def deny_read(path, *args, **kwargs):
        if path == source:
            raise PermissionError("denied by test")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_read)
    client = CaptureModel()
    assert_blocked(make_agent(tmp_path, client), client, "read")


def test_outside_workspace_project_rules_block_request(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"outside project marker")
    source = workspace / "AGENTS.md"
    original_resolve = Path.resolve
    try:
        source.symlink_to(outside)
    except OSError as error:
        if getattr(error, "winerror", None) not in {5, 1314}:
            raise
        # Simulate only symlink resolution if this Windows account cannot make one.
        # Workspace.resolve_path still performs the real containment check.
        source.touch()

        def resolve_test_link(path, *args, **kwargs):
            target = outside if path == source else path
            return original_resolve(target, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve_test_link)
    original_open = Path.open

    def forbid_outside_open(path, *args, **kwargs):
        assert path not in {source, outside}, "escaping project rules must not be read"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_outside_open)
    client = CaptureModel()
    assert_blocked(make_agent(workspace, client), client, "outside")


def test_replay_never_reads_project_rules_and_edits_leave_log_bytes_unchanged(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rules(workspace, "Initially visible only to model.")
    store = JsonlEventStore(tmp_path / "history")
    with SessionService(store).create(workspace) as handle:
        session_id = handle.session_id
        client = CaptureModel()
        agent = make_agent(workspace, client, writer=handle.writer)
        assert agent.run_turn("saved task").status == "completed"
        assert_project_background(client.requests[0], "Initially visible only to model.")
    path = tmp_path / "history" / "sessions" / f"{session_id}.jsonl"
    log = path.read_bytes()
    assert b"Initially visible only to model." not in log
    original_open = Path.open

    def forbid_project_rules(path, *args, **kwargs):
        if path.name == "AGENTS.md":
            pytest.fail("Replay must not read project data.")
        return original_open(path, *args, **kwargs)

    replay = SessionReplay(store)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbid_project_rules)
        before = replay.replay(session_id, "fixed")
    rules(workspace, "New manual project data.")
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbid_project_rules)
        assert replay.replay(session_id, "fixed") == before
    assert path.read_bytes() == log
