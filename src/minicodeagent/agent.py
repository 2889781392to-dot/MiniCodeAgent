"""Coordinate one synchronous model/tool turn."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from .chat_payload import encode_chat_payload
from .config import LoopLimits
from .context.budget import BudgetPolicy, ContextMeasurementSnapshot, SystemPromptParts, TokenEstimate, TokenEstimator
from .context.builder import ContextBuilder, ContextProjection
from .context.events import (
    AssistantMessagePayload,
    HistoryError,
    ModelRequestStartedPayload,
    ModelRequestTrigger,
    SessionCreatedPayload,
    ToolResultPayload,
    TurnFinishedPayload,
    UserMessagePayload,
    inspect_history,
)
from .context.manager import CompactionDiagnostic, ContextWindowManager
from .context.replay import SessionState
from .context.writer import EventWriteError, SessionEventWriter
from .project_memory import ProjectMemoryError, load_project_memory
from .protocol import ChatRequest, ModelError, ModelReply, TurnOutcome
from .tools.registry import ToolRegistry
from .workspace import Workspace

SYSTEM_PROMPT = (
    "You are a coding agent in a local project. Inspect relevant files before editing, "
    "and use the tools to verify changes when possible. Commands require user approval. "
    "Use argv with a workspace cwd. Preserve unrelated files and changes. "
    "Report what you changed, what was actually tested, and any remaining issues."
)


class ModelClient(Protocol):
    def complete(self, request: ChatRequest) -> ModelReply: ...


@dataclass
class _Turn:
    id: UUID = field(default_factory=uuid4)
    model_attempts: int = 0
    tool_batches: int = 0
    retried_transient: bool = False
    retried_overflow: bool = False
    in_flight_call: UUID | None = None
    request_id: UUID | None = None


class AgentLoop:
    def __init__(
        self,
        client: ModelClient,
        registry: ToolRegistry,
        workspace: Workspace,
        writer: SessionEventWriter,
        builder: ContextBuilder,
        limits: LoopLimits,
        *,
        track_requests: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
        environment_prompt: Callable[[], str] | None = None,
        token_estimator: TokenEstimator | None = None,
        budget_policy: BudgetPolicy | None = None,
        compact_context: bool = False,
        on_measurement: Callable[[ContextMeasurementSnapshot], None] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.workspace = workspace
        self.writer = writer
        self.builder = builder
        self.limits = limits
        self.track_requests = track_requests
        self.system_prompt = system_prompt
        self._environment_prompt = environment_prompt
        self._token_estimator = token_estimator if token_estimator is not None else TokenEstimator()
        self._budget_policy = budget_policy
        self._window_manager = None
        if compact_context:
            if budget_policy is None:
                raise ValueError("Context compaction requires an explicit budget policy.")
            payload_encoder = getattr(type(client), "request_payload", None)
            encode = client.request_payload if payload_encoder is not None else encode_chat_payload
            summarizer = client.summarize if getattr(type(client), "summarize", None) is not None else None
            encode_summary = client.summary_payload if getattr(type(client), "summary_payload", None) is not None else None
            self._window_manager = ContextWindowManager(
                builder, self._token_estimator, budget_policy, encode=encode, enable_checkpoint=True,
                summarizer=summarizer, encode_summary=encode_summary,
            )
        self._on_measurement = on_measurement
        self._measurements: list[ContextMeasurementSnapshot] = []

    @property
    def measurements(self) -> tuple[ContextMeasurementSnapshot, ...]:
        """Local observations for the most recent run_turn or continue_turn."""
        return tuple(self._measurements)

    def run_turn(self, content: str) -> TurnOutcome:
        self._measurements.clear()
        turn = _Turn()
        if not content.strip():
            return TurnOutcome(turn_id=turn.id, status="failed", reason="empty_input")
        return self._run(turn, trigger="initial", content=content)

    def continue_turn(self, state: SessionState) -> TurnOutcome:
        self._measurements.clear()
        if state.session_id != self.writer.session_id or state.history != self.writer.history:
            raise HistoryError("Continuation requires the current session history.")
        current = state.current_turn
        if current is None or current.finished is not None:
            raise HistoryError("There is no open turn to continue.")
        if current.pending_calls:
            raise HistoryError("Settle pending calls before continuing the turn.")
        if not self.track_requests or current.latest_request is None:
            raise HistoryError("Request attempt history is unavailable. Start a new turn.")
        reply = current.latest_reply.payload if current.latest_reply is not None else None
        if (
            isinstance(reply, AssistantMessagePayload)
            and reply.request_id == current.latest_request.request_id
            and not reply.tool_calls
        ):
            raise HistoryError("The latest reply must be finalized before continuing.")
        turn = _Turn(
            id=current.turn_id, model_attempts=current.model_attempts,
            tool_batches=current.tool_batches, retried_transient=current.retried_transient,
            retried_overflow=current.retried_overflow,
        )
        return self._run(turn, trigger="resume")

    def _run(self, turn: _Turn, *, trigger: ModelRequestTrigger, content: str | None = None) -> TurnOutcome:
        try:
            try:
                system_parts = (
                    self._request_system_prompt()
                    if turn.model_attempts < self.limits.max_provider_calls else SystemPromptParts(self.system_prompt)
                )
                if content is not None:
                    self.writer.append("user_message", UserMessagePayload(turn_id=turn.id, content=content))
                outcome = self._drive(turn, trigger, system_parts)
            except KeyboardInterrupt:
                active, pending = inspect_history(self.writer.history)
                if active == turn.id:
                    for call in pending:
                        started = call.call_id == turn.in_flight_call
                        self.writer.append("tool_result", ToolResultPayload(
                            turn_id=turn.id,
                            call_id=call.call_id,
                            status="interrupted",
                            output="Interrupted. The execution result is unknown." if started
                            else "Skipped because the turn was interrupted before dispatch.",
                            details={"execution_outcome": "unknown" if started else "not_started"},
                        ))
                outcome = TurnOutcome(turn_id=turn.id, status="interrupted", reason="interrupted")
            except HistoryError as exc:
                outcome = TurnOutcome(turn_id=turn.id, status="failed", reason="context_error", content=str(exc))

            if any(
                isinstance(event.payload, UserMessagePayload) and event.payload.turn_id == turn.id
                for event in self.writer.history
            ):
                outcome = self._finish_turn(outcome)
            return outcome
        except ProjectMemoryError as exc:
            return TurnOutcome(turn_id=turn.id, status="failed", reason="project_memory_error", content=str(exc))
        except EventWriteError as exc:
            # Do not dispatch further work, or claim the turn was committed.
            return TurnOutcome(turn_id=turn.id, status="failed", reason="writer_error", content=str(exc))

    def _request_system_prompt(self) -> SystemPromptParts:
        history = self.writer.history
        metadata = history[0].payload if history else None
        project_memory = ""
        if not isinstance(metadata, SessionCreatedPayload) or Path(metadata.workspace) == self.workspace.root:
            background = load_project_memory(self.workspace)
            if background is not None:
                project_memory = (
                    "\n\n[Project background from Workspace AGENTS.md]\n"
                    f"{background}\n"
                    "[End project background]"
                )
        runtime_context = "\n\n" + self._environment_prompt() if self._environment_prompt is not None else ""
        return SystemPromptParts(self.system_prompt, project_memory, runtime_context)

    def _drive(self, turn: _Turn, trigger: ModelRequestTrigger, system_parts: SystemPromptParts) -> TurnOutcome:
        system_prompt = system_parts.text
        while turn.model_attempts < self.limits.max_provider_calls:
            history = self.writer.history
            projection = self.builder.project(history, system_prompt)
            with self.registry.fixed_bindings():
                tools = self.registry.definitions()
                comparison = None
                if self._window_manager is None:
                    request = ChatRequest(messages=projection.messages, tools=tools)
                else:
                    prepared = self._window_manager.prepare(history, projection, system_parts, tools)
                    comparison = prepared.diagnostic
                    if not comparison.can_send:
                        assert self._budget_policy is not None
                        return TurnOutcome(
                            turn_id=turn.id, status="limited", reason="context_budget",
                            content=(
                                f"Context budget [{comparison.status}]. "
                                f"Original estimate {comparison.original_tokens}, "
                                f"candidate estimate {comparison.candidate_tokens}, "
                                f"input capacity {self._budget_policy.input_capacity}."
                            ),
                        )
                    if prepared.compaction is not None:
                        self.writer.append("compaction_committed", prepared.compaction)
                    request, projection = prepared.request, prepared.projection
                try:
                    reply, projection = self._request_with_overflow_recovery(
                        turn, request, projection, trigger, system_parts, comparison,
                    )
                except ModelError as exc:
                    return TurnOutcome(turn_id=turn.id, status="failed", reason=exc.kind, content=str(exc))

                if reply.finish_reason == "length":
                    seq = self._save_reply(turn, reply, projection)
                    return TurnOutcome(turn_id=turn.id, status="limited", reason="output_length",
                                       content=reply.content, assistant_seq=seq)
                if reply.finish_reason == "content_filter":
                    return TurnOutcome(turn_id=turn.id, status="failed", reason="content_filter")

                if reply.tool_calls or reply.finish_reason == "tool_calls":
                    reason = self._batch_rejection(turn, reply, projection)
                    if reason == "protocol_error":
                        return TurnOutcome(
                            turn_id=turn.id, status="failed", reason=reason,
                            content="The model returned an incomplete or ambiguous tool batch.",
                        )
                    if reason is not None:
                        seq = self._save_reply(turn, reply, projection, rejected=True)
                        return TurnOutcome(turn_id=turn.id, status="limited", reason=reason,
                                           content=reply.content, assistant_seq=seq)

                    self._save_reply(turn, reply, projection, accept=True)
                    turn.tool_batches += 1
                    for call in reply.tool_calls:
                        turn.in_flight_call = call.call_id
                        result = self.registry.execute(call, self.workspace)
                        self.writer.append("tool_result", ToolResultPayload(
                            turn_id=turn.id, **result.model_dump(mode="python"),
                        ))
                        turn.in_flight_call = None
                        if result.status == "interrupted":
                            raise KeyboardInterrupt
                    trigger = "after_tools"
                    continue

                seq = self._save_reply(turn, reply, projection)
                completed = reply.finish_reason == "stop" and bool(reply.content.strip())
                return TurnOutcome(
                    turn_id=turn.id, status="completed" if completed else "failed",
                    reason="model_stop" if completed else "unexpected_finish_reason",
                    content=reply.content, assistant_seq=seq,
                )

        return TurnOutcome(turn_id=turn.id, status="limited", reason="provider_call_limit")

    def _request_with_overflow_recovery(
        self, turn: _Turn, request: ChatRequest, projection: ContextProjection,
        trigger: ModelRequestTrigger, system_parts: SystemPromptParts, comparison: CompactionDiagnostic | None,
    ) -> tuple[ModelReply, ContextProjection]:
        try:
            return self._request(turn, request, trigger, system_parts, comparison), projection
        except ModelError as exc:
            if exc.kind != "context_length" or self._window_manager is None or turn.retried_overflow:
                raise

        if turn.model_attempts >= self.limits.max_provider_calls:
            raise ModelError("context_length", "The provider rejected the context length. No main attempts remain.")
        observation = self._measurements[-1] if self._measurements else None
        if observation is None or observation.request_id != turn.request_id or (
            rejected_tokens := observation.estimate.estimated_input_tokens
        ) is None:
            raise ModelError("context_length", "The rejected request could not be measured for safe recovery.")

        prepared = self._window_manager.prepare(
            self.writer.history, projection, system_parts, request.tools, rejected_tokens=rejected_tokens,
        )
        if not prepared.diagnostic.can_send or prepared.compaction is None:
            raise ModelError(
                "context_length", f"No strictly smaller context fits after provider rejection "
                f"[{prepared.diagnostic.status}; rejected estimate {rejected_tokens}].",
            )
        self.writer.append("compaction_committed", prepared.compaction)
        turn.retried_overflow = True
        reply = self._request(turn, prepared.request, "retry_overflow", system_parts, prepared.diagnostic)
        return reply, prepared.projection

    def _request(
        self, turn: _Turn, request: ChatRequest, trigger: ModelRequestTrigger, system_parts: SystemPromptParts,
        comparison: CompactionDiagnostic | None,
    ) -> ModelReply:
        while True:
            start = ModelRequestStartedPayload(
                turn_id=turn.id, request_id=uuid4(), attempt=turn.model_attempts + 1, trigger=trigger,
            )
            if self.track_requests:
                self.writer.append("model_request_started", start)
            turn.request_id = start.request_id
            turn.model_attempts = start.attempt
            try:
                return self._complete(request, system_parts, start, comparison)
            except ModelError as exc:
                if (trigger == "retry_overflow" or not exc.retryable or turn.retried_transient
                        or turn.model_attempts >= self.limits.max_provider_calls):
                    raise
                turn.retried_transient = True
                trigger = "retry_transient"

    def _complete(
        self, request: ChatRequest, system_parts: SystemPromptParts, start: ModelRequestStartedPayload,
        comparison: CompactionDiagnostic | None,
    ) -> ModelReply:
        snapshot = None
        reply = None
        failure = None

        def measure(payload: dict | None = None) -> None:
            nonlocal snapshot
            try:
                estimate = self._token_estimator.estimate(request, system_parts=system_parts, payload=payload)
            except Exception:
                estimate = TokenEstimator.unavailable(request, "Tokenizer could not measure this request.")
            if not isinstance(estimate, TokenEstimate):
                estimate = TokenEstimator.unavailable(request, "Tokenizer returned an invalid measurement.")
            snapshot = ContextMeasurementSnapshot(
                request_id=start.request_id, turn_id=start.turn_id, attempt=start.attempt, trigger=start.trigger,
                estimate=estimate, budget_policy=self._budget_policy,
                dispatch_boundary="model_client" if payload is None else "sdk_create",
                compaction=comparison,
            )

        try:
            # The optional adapter method supplies the very same payload used for SDK dispatch.
            complete_with_payload = getattr(type(self.client), "complete_with_payload", None)
            if complete_with_payload is None:
                measure()
                response = self.client.complete(request)
            else:
                response = complete_with_payload(self.client, request, measure)
            if isinstance(response, ModelReply):
                # Pydantic copies and nested model instances can bypass validation.
                try:
                    reply = ModelReply.model_validate(response.model_dump(warnings=False))
                except ValueError:
                    pass
                else:
                    return reply
            raise ModelError("protocol_error", "The model client returned an invalid reply.")
        except ModelError as exc:
            failure = exc.kind
            raise
        except KeyboardInterrupt:
            failure = "interrupted"
            raise
        except Exception:
            failure = "client_error"
            raise
        finally:
            if snapshot is not None:
                observation = replace(
                    snapshot,
                    actual_usage=reply.usage if reply is not None else None,
                    prompt_tokens_details=reply.prompt_tokens_details if reply is not None else None,
                    failure_kind=failure,
                )
                self._measurements.append(observation)
                if self._on_measurement is not None:
                    try:
                        self._on_measurement(observation)
                    except Exception:
                        # An observer does not participate in request or event settlement.
                        pass

    def _finish_turn(self, outcome: TurnOutcome) -> TurnOutcome:
        payload = TurnFinishedPayload(
            turn_id=outcome.turn_id, status=outcome.status, reason=outcome.reason,
            assistant_seq=outcome.assistant_seq,
        )
        try:
            self.writer.append("turn_finished", payload)
        except KeyboardInterrupt:
            active, _ = inspect_history(self.writer.history)
            if active == outcome.turn_id:
                # No terminal event was committed. Close the turn once as interrupted.
                self.writer.append("turn_finished", payload.model_copy(update={
                    "status": "interrupted", "reason": "interrupted",
                }))
                return TurnOutcome(
                    turn_id=outcome.turn_id, status="interrupted", reason="interrupted",
                    assistant_seq=outcome.assistant_seq,
                )
        return outcome

    def _batch_rejection(
        self, turn: _Turn, reply: ModelReply, projection: ContextProjection,
    ) -> str | None:
        calls = reply.tool_calls
        local_ids = {call.call_id for call in calls}
        provider_ids = {call.provider_call_id for call in calls}
        if (
            reply.finish_reason != "tool_calls" or not calls or len(local_ids) != len(calls)
            or len(provider_ids) != len(calls) or local_ids.intersection(projection.consumed_call_ids)
        ):
            return "protocol_error"
        if len(calls) > self.limits.max_calls_per_batch:
            return "tool_batch_size_limit"
        if turn.tool_batches >= self.limits.max_tool_batches:
            return "tool_batch_limit"
        if turn.model_attempts >= self.limits.max_provider_calls:
            return "provider_call_limit"
        return None

    def _save_reply(
        self, turn: _Turn, reply: ModelReply, projection: ContextProjection,
        *, accept: bool = False, rejected: bool = False,
    ) -> int:
        assert turn.request_id is not None
        event = self.writer.append("assistant_message", AssistantMessagePayload(
            turn_id=turn.id,
            request_id=turn.request_id,
            content=reply.content,
            finish_reason=reply.finish_reason,
            usage=reply.usage,
            tool_calls=reply.tool_calls if accept else (),
            rejected_tool_calls=reply.tool_calls if rejected else (),
            consumed_call_ids=projection.consumed_call_ids,
        ))
        return event.seq
