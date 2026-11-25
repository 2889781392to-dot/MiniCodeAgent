"""Assemble the agent and display committed activity in a terminal."""

import argparse
from collections.abc import Callable
from dataclasses import asdict
from functools import cache
import json
import os
import sys
from pathlib import Path
from typing import TextIO
from uuid import UUID

from .agent import SYSTEM_PROMPT, AgentLoop
from .config import LoopLimits, Settings
from .context.budget import BudgetPolicy, ContextMeasurementSnapshot
from .context.builder import ContextBuilder
from .context.events import AssistantMessagePayload, SessionEvent, ToolResultPayload
from .context.store import EventStoreError, JsonlEventStore
from .context.writer import EventWriteError, SessionEventWriter
from .llm import LLMClient
from .protocol import ToolCall, TurnOutcome
from .python_environment import discover_python_environment
from .session.catalog import SessionCatalogError
from .session.service import SessionService
from .tools.builtin import register_builtin_tools
from .tools.registry import ToolRegistry
from .workspace import Workspace


def _print(text: str, stream: TextIO) -> None:
    text = text.replace("\r\n", "\n")
    safe = "".join(
        char if char.isprintable() or char in "\n\t" else ascii(char)[1:-1]
        for char in text
    )
    encoding = getattr(stream, "encoding", None) or "utf-8"
    print(safe.encode(encoding, "backslashreplace").decode(encoding), file=stream, flush=True)


def confirm_command(argv: tuple[str, ...], cwd: Path) -> bool:
    _print(f"Command cwd: {json.dumps(str(cwd), ensure_ascii=False)}", sys.stderr)
    _print(f"Command argv: {json.dumps(argv, ensure_ascii=False)}", sys.stderr)
    if not sys.stdin.isatty():
        _print("Interactive confirmation is unavailable. Command not started.", sys.stderr)
        return False
    try:
        return input("Run this command once? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _show_event(event: SessionEvent) -> None:
    payload = event.payload
    if isinstance(payload, AssistantMessagePayload):
        for call in payload.tool_calls:
            _print(f"[call {call.call_id.hex[:8]}] {call.name[:100]}", sys.stdout)
    elif isinstance(payload, ToolResultPayload):
        preview = payload.output[:1_200]
        if len(preview) < len(payload.output):
            preview += "\n[terminal preview truncated]"
        _print(f"[result {payload.call_id.hex[:8]}] {payload.status}\n{preview}", sys.stdout)


def _show_measurement(snapshot: ContextMeasurementSnapshot) -> None:
    data = asdict(snapshot)
    data["estimate"]["status"] = snapshot.estimate.status
    if snapshot.estimate.breakdown is not None:
        data["estimate"]["breakdown"]["history_tokens"] = snapshot.estimate.breakdown.history_tokens
    if snapshot.compaction is not None and snapshot.compaction.archive is not None:
        data["compaction"]["archive"]["covered_events"] = snapshot.compaction.archive.covered_events
    for key in ("actual_usage", "prompt_tokens_details"):
        value = getattr(snapshot, key)
        data[key] = value.model_dump(exclude_none=True) if value is not None else None
    data.update(
        usage_status=snapshot.usage_status,
        prompt_token_delta=snapshot.prompt_token_delta,
        at_or_above_high_watermark=snapshot.at_or_above_high_watermark,
        at_or_above_input_capacity=snapshot.at_or_above_input_capacity,
    )
    _print("[context measurement] " + json.dumps(data, default=str), sys.stderr)


def _show_outcome(outcome: TurnOutcome) -> int:
    if outcome.content:
        _print(outcome.content, sys.stderr if outcome.status == "failed" else sys.stdout)
    _print(f"[{outcome.status}] {outcome.reason}", sys.stdout)
    return {"completed": 0, "limited": 1, "failed": 1, "interrupted": 130}[outcome.status]


def run_repl(agent: AgentLoop) -> None:
    while True:
        try:
            task = input("MiniCodeAgent> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if task in {"/exit", "/quit"}:
            return
        if not task:
            continue
        outcome = agent.run_turn(task)
        _show_outcome(outcome)
        if outcome.reason == "writer_error":
            return


def _confirm_recovery(calls: tuple[ToolCall, ...]) -> bool:
    _print("These calls have no recorded result. Their execution outcomes are unknown.", sys.stderr)
    for call in calls:
        _print(f"[unknown {call.call_id}] {call.name}", sys.stderr)
    if not sys.stdin.isatty():
        return False
    try:
        return input("Record these outcomes as interrupted and unknown? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _python_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("expected an absolute Python executable path")
    return path


def _add_context_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--measure-context", action="store_true", help="Display per-attempt context observations.")
    parser.add_argument(
        "--compact-context", nargs=2, type=int, metavar=("WINDOW", "OUTPUT_RESERVE"),
        help="Enable L1 through L4 context compaction with an explicit token window and output reserve.",
    )


def _budget_policy(parser: argparse.ArgumentParser, window_and_reserve: list[int] | None) -> BudgetPolicy | None:
    if window_and_reserve is None:
        return None
    try:
        return BudgetPolicy(*window_and_reserve)
    except ValueError as exc:
        parser.error(str(exc))


def _run_environment_prompt(workspace: Workspace, explicit_python: Path | None) -> Callable[[], str]:
    @cache
    def snapshot() -> str:
        observation = discover_python_environment(
            workspace, explicit_python=explicit_python, confirm=confirm_command,
        )
        prompt = observation.prompt()
        _print(prompt, sys.stderr)
        return prompt

    return snapshot


def _resume_command(service: SessionService, options: argparse.Namespace) -> int:
    with service.resume(
        options.session_id, workspace=options.workspace, continue_turn=options.continue_turn,
        confirm_pending=_confirm_recovery,
    ) as handle:
        recovery = handle.replay_result
        if recovery.unknown_call_ids:
            ids = ", ".join(map(str, sorted(recovery.unknown_call_ids)))
            _print(f"Unknown tool outcomes ({ids}). Verify effects before retrying.", sys.stderr)
        prompt = SYSTEM_PROMPT
        metadata = recovery.state.metadata
        assert metadata is not None
        if handle.workspace.root != Path(metadata.workspace):
            guidance = (
                "The current workspace differs from the saved workspace. "
                "Re-read files and verify prior observations in the current workspace before relying on them."
            )
            _print(f"{guidance} Project Memory is paused for this workspace override.", sys.stderr)
            prompt += " " + guidance
        if handle.recovered_reply is not None:
            payload = handle.recovered_reply.payload
            turn = recovery.state.current_turn
            assert isinstance(payload, AssistantMessagePayload) and turn is not None and turn.finished is not None
            code = _show_outcome(TurnOutcome(
                turn_id=turn.turn_id, status=turn.finished.status, reason=turn.finished.reason,
                content=payload.content, assistant_seq=turn.finished.assistant_seq,
            ))
            if options.continue_turn:
                return code

        settings = Settings.from_env(os.environ)
        registry = ToolRegistry()
        register_builtin_tools(registry, confirm_command=confirm_command)
        client = LLMClient(settings)
        try:
            agent = AgentLoop(
                client, registry, handle.workspace, handle.writer, ContextBuilder(), settings.limits,
                track_requests=True, system_prompt=prompt,
                environment_prompt=_run_environment_prompt(handle.workspace, options.python),
                budget_policy=options.budget_policy, compact_context=options.budget_policy is not None,
                on_measurement=_show_measurement if options.measure_context else None,
            )
            if options.continue_turn:
                return _show_outcome(agent.continue_turn(recovery.state))
            if options.message is not None:
                return _show_outcome(agent.run_turn(options.message))
            run_repl(agent)
            return 0
        finally:
            client.close()


def _session_command(command: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"minicodeagent {command}")
    parser.add_argument("--data-root", type=Path, help="Session data directory (default: ~/.minicodeagent).")
    if command == "new":
        parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project directory.")
        parser.add_argument("--title", help="Defaults to the project directory name.")
    elif command in {"resume", "fork"}:
        parser.add_argument("session_id", type=UUID)
        parser.add_argument("--workspace", type=Path, help="Workspace override.")
        if command == "fork":
            parser.add_argument("--title", help="Title for the child session.")
        else:
            parser.add_argument("--python", type=_python_path, help="Absolute Python executable for this run.")
            _add_context_options(parser)
            mode = parser.add_mutually_exclusive_group()
            mode.add_argument("--message", help="Close the old turn if needed and start a new one.")
            mode.add_argument("--continue", dest="continue_turn", action="store_true", help="Continue the open turn.")
    options = parser.parse_args(argv)
    if command == "resume":
        options.budget_policy = _budget_policy(parser, options.compact_context)
    if command == "resume" and options.message is not None and not options.message.strip():
        parser.error("--message cannot be blank")
    try:
        root = options.data_root if options.data_root is not None else Path.home() / ".minicodeagent"
        service = SessionService(JsonlEventStore(root), system_prompt=SYSTEM_PROMPT, on_event=_show_event)
        if command == "new":
            with service.create(options.workspace, options.title) as handle:
                session_id = handle.session_id
            _print(str(session_id), sys.stdout)
        elif command == "fork":
            with service.fork(options.session_id, workspace=options.workspace, title=options.title) as handle:
                child_id = handle.session_id
            _print(str(child_id), sys.stdout)
        elif command == "resume":
            return _resume_command(service, options)
        else:
            records = [record.model_dump(mode="json") for record in service.list()]
            _print(json.dumps(records, ensure_ascii=False, indent=2), sys.stdout)
    except (ValueError, EventStoreError, EventWriteError, SessionCatalogError) as exc:
        _print(str(exc), sys.stderr)
        return 2
    except KeyboardInterrupt:
        _print("[interrupted] interrupted", sys.stdout)
        return 130
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in {"new", "list", "resume", "fork"}:
        return _session_command(args[0], args[1:])
    parser = argparse.ArgumentParser(
        description="A local coding agent with six workspace tools.",
        epilog="Session commands are new, list, resume and fork. Use a command with --help for options.",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project directory (default: current directory).")
    parser.add_argument("--python", type=_python_path, help="Absolute Python executable for this run.")
    _add_context_options(parser)
    parser.add_argument("--message", help="Run one task and exit. Omit to enter the REPL.")
    parser.add_argument("--max-provider-calls", type=int, help="Model attempt limit per turn (default: 12).")
    parser.add_argument("--max-tool-batches", type=int, help="Tool batch limit per turn (default: 8).")
    parser.add_argument("--max-calls-per-batch", type=int, help="Tool call limit per batch (default: 8).")
    options = parser.parse_args(args)
    budget_policy = _budget_policy(parser, options.compact_context)
    if options.message is not None and not options.message.strip():
        parser.error("--message cannot be blank")

    try:
        settings = Settings.from_env(os.environ)
        limits = LoopLimits(**{
            name: getattr(options, name) if getattr(options, name) is not None else getattr(settings.limits, name)
            for name in ("max_provider_calls", "max_tool_batches", "max_calls_per_batch")
        })
        workspace = Workspace(options.workspace)
    except ValueError as exc:
        _print(str(exc), sys.stderr)
        return 2

    registry = ToolRegistry()
    register_builtin_tools(registry, confirm_command=confirm_command)
    writer = SessionEventWriter(on_event=_show_event)
    client = LLMClient(settings)
    try:
        agent = AgentLoop(
            client, registry, workspace, writer, ContextBuilder(), limits,
            environment_prompt=_run_environment_prompt(workspace, options.python),
            budget_policy=budget_policy, compact_context=budget_policy is not None,
            on_measurement=_show_measurement if options.measure_context else None,
        )
        if options.message is not None:
            return _show_outcome(agent.run_turn(options.message))
        run_repl(agent)
        return 0
    except KeyboardInterrupt:
        _print("[interrupted] interrupted", sys.stdout)
        return 130
    finally:
        client.close()
