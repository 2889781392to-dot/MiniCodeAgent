"""Measure a Chat Completions request without a provider tokenizer or SDK."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from ..chat_payload import encode_chat_payload
from ..protocol import ChatRequest, PromptTokenDetails, TokenUsage

if TYPE_CHECKING:
    from .manager import CompactionDiagnostic


_TOKENIZER_ID = "utf8-compact-json-heuristic"
_TOKENIZER_VERSION = "1"
_BYTES_PER_TOKEN = 3
_FRAME_TOKENS = 3
_MAX_ESTIMATION_BYTES = 4 * 1024 * 1024
_JSON_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
_CONTENT_COMPONENT = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool_result"}
_SYSTEM_COMPONENTS = ("system", "project_memory", "runtime_context")


@dataclass(frozen=True, slots=True)
class SystemPromptParts:
    """Exact segments of the emitted system message, including their separators."""

    system_prompt: str
    project_memory: str = ""
    runtime_context: str = ""

    @property
    def text(self) -> str:
        return self.system_prompt + self.project_memory + self.runtime_context


@dataclass(frozen=True, slots=True)
class TokenBreakdown:
    system_tokens: int
    project_memory_tokens: int
    runtime_context_tokens: int
    user_tokens: int
    assistant_tokens: int
    tool_result_tokens: int
    tool_definition_tokens: int
    other_metadata_tokens: int
    unclassified_tokens: int

    @property
    def history_tokens(self) -> int:
        return self.user_tokens + self.assistant_tokens + self.tool_result_tokens


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    message_count: int
    estimated_input_tokens: int | None = None
    breakdown: TokenBreakdown | None = None
    tokenizer_id: str = _TOKENIZER_ID
    tokenizer_version: str = _TOKENIZER_VERSION
    diagnostic: str | None = None

    @property
    def status(self) -> Literal["estimated", "unavailable"]:
        return "unavailable" if self.estimated_input_tokens is None else "estimated"


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Optional context-window reference values supplied by the caller."""

    model_context_window: int
    output_reserve: int

    def __post_init__(self) -> None:
        if type(self.model_context_window) is not int or self.model_context_window <= 0:
            raise ValueError("The model context window must be a positive integer.")
        if type(self.output_reserve) is not int or self.output_reserve < 0:
            raise ValueError("The output reserve must be a nonnegative integer.")
        if self.input_capacity <= 0:
            raise ValueError("The remaining input capacity must be positive.")

    @property
    def input_capacity(self) -> int:
        return self.model_context_window * 95 // 100 - self.output_reserve

    @property
    def high_watermark(self) -> int:
        return self.input_capacity * 85 // 100

    @property
    def low_watermark(self) -> int:
        return self.input_capacity * 70 // 100


@dataclass(frozen=True, slots=True)
class ContextMeasurementSnapshot:
    request_id: UUID
    turn_id: UUID
    attempt: int
    trigger: str
    estimate: TokenEstimate
    actual_usage: TokenUsage | None = None
    prompt_tokens_details: PromptTokenDetails | None = None
    budget_policy: BudgetPolicy | None = None
    dispatch_boundary: Literal["model_client", "sdk_create"] = "model_client"
    failure_kind: str | None = None
    compaction: CompactionDiagnostic | None = None

    @property
    def usage_status(self) -> Literal["available", "unavailable"]:
        return "unavailable" if self.actual_usage is None else "available"

    @property
    def prompt_token_delta(self) -> int | None:
        estimated = self.estimate.estimated_input_tokens
        if self.actual_usage is None or estimated is None:
            return None
        return self.actual_usage.input_tokens - estimated

    @property
    def at_or_above_high_watermark(self) -> bool | None:
        estimated = self.estimate.estimated_input_tokens
        if self.budget_policy is None or estimated is None:
            return None
        return estimated >= self.budget_policy.high_watermark

    @property
    def at_or_above_input_capacity(self) -> bool | None:
        estimated = self.estimate.estimated_input_tokens
        if self.budget_policy is None or estimated is None:
            return None
        return estimated >= self.budget_policy.input_capacity


class _Oversized(ValueError):
    pass


def _rounded_tokens(byte_count: int) -> int:
    return (byte_count + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN


def _scalar_json_bytes(value: object) -> int:
    return len(_JSON_ENCODER.encode(value).encode("utf-8"))


def _compact_json_size(value: object, limit: int) -> int:
    """Count exact compact JSON bytes with a work bound and no full JSON string."""
    remaining = limit

    def spend(amount: int) -> None:
        nonlocal remaining
        remaining -= amount
        if remaining < 0:
            raise _Oversized

    def visit(item: object) -> None:
        if isinstance(item, dict):
            spend(2 + len(item) + max(0, len(item) - 1))  # braces, colons and commas
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("Expected a string JSON key.")
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            spend(2 + max(0, len(item) - 1))  # brackets and commas
            for child in item:
                visit(child)
        else:
            if isinstance(item, str) and len(item) + 2 > remaining:
                raise _Oversized
            spend(_scalar_json_bytes(item))

    visit(value)
    return limit - remaining


class TokenEstimator:
    """Repeatable, provider-independent estimate based on emitted JSON and framing."""

    def __init__(self, *, max_bytes: int = _MAX_ESTIMATION_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("The measurement byte limit must be a positive integer.")
        self._max_bytes = max_bytes

    def estimate_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise ValueError("Expected text as a string.")
        if len(text) <= self._max_bytes:
            try:
                size = len(text.encode("utf-8"))
            except UnicodeError:
                raise ValueError("Text cannot be encoded as UTF-8.") from None
            if size <= self._max_bytes:
                return _rounded_tokens(size)
        raise ValueError("Text exceeds the measurement byte limit.")

    @staticmethod
    def unavailable(request: ChatRequest, diagnostic: str) -> TokenEstimate:
        count = len(request.messages) if isinstance(request, ChatRequest) and isinstance(request.messages, tuple) else 0
        return TokenEstimate(message_count=count, diagnostic=diagnostic)

    def estimate(
        self, request: ChatRequest, *, system_parts: SystemPromptParts | None = None, payload: dict | None = None,
    ) -> TokenEstimate:
        if not isinstance(request, ChatRequest):
            return self.unavailable(request, "Expected a ChatRequest for measurement.")
        try:
            if payload is None:
                self._check_projection_size(request)
                payload = encode_chat_payload(request)
            if not isinstance(payload, dict) or not isinstance(payload.get("extra_body", {}), dict):
                return self.unavailable(request, "Expected a JSON request payload and optional extra_body object.")
            extras = payload.get("extra_body", {})
            if len(payload) + len(extras) > self._max_bytes:
                raise _Oversized
            body = {key: value for key, value in payload.items() if key != "extra_body"}
            body.update(extras)
            messages = body.get("messages")
            if not isinstance(messages, list) or len(messages) != len(request.messages):
                return self.unavailable(request, "Wire messages do not match the ChatRequest.")

            body_bytes = _compact_json_size(body, self._max_bytes)
            counts = self._component_bytes(body, request, messages, system_parts)
            estimates = {f"{key}_tokens": size // _BYTES_PER_TOKEN for key, size in counts.items()}
            total = _rounded_tokens(body_bytes) + _FRAME_TOKENS * (len(messages) + 1)
            breakdown = TokenBreakdown(unclassified_tokens=total - sum(estimates.values()), **estimates)
            return TokenEstimate(len(messages), total, breakdown)
        except _Oversized:
            return self.unavailable(request, f"Request exceeds the {self._max_bytes}-byte measurement limit.")
        except UnicodeError:
            return self.unavailable(request, "Request contains text that cannot be encoded as UTF-8.")
        except RecursionError:
            return self.unavailable(request, "Request JSON is too deeply nested for measurement.")
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return self.unavailable(request, "Request has invalid JSON values or mismatched message sources.")

    def _check_projection_size(self, request: ChatRequest) -> None:
        # Bound work before rendering fallback wire calls, including parsing their arguments.
        size = len(request.tools) + len(request.messages)
        if size > self._max_bytes:
            raise _Oversized
        for message in request.messages:
            size += len(message.content) + len(message.tool_calls)
            if size > self._max_bytes:
                raise _Oversized
            for call in message.tool_calls:
                size += len(call.arguments_raw)
                if size > self._max_bytes:
                    raise _Oversized

    def _component_bytes(
        self, body: dict, request: ChatRequest, messages: list, system_parts: SystemPromptParts | None,
    ) -> dict[str, int]:
        if not messages and system_parts is not None:
            raise ValueError("System prompt segments require a system message.")
        counts = dict.fromkeys((*_SYSTEM_COMPONENTS, "user", "assistant", "tool_result",
                                "tool_definition", "other_metadata"), 0)
        for index, (message, original) in enumerate(zip(messages, request.messages, strict=True)):
            role, content = message["role"], message["content"]
            if role != original.role or content != original.content:
                raise ValueError("The wire message differs from the ChatRequest.")
            if index == 0 and system_parts is not None:
                pieces = (system_parts.system_prompt, system_parts.project_memory, system_parts.runtime_context)
                if role != "system" or sum(map(len, pieces)) != len(content) or system_parts.text != content:
                    raise ValueError("System prompt segments do not match the wire content.")
                for name, piece in zip(_SYSTEM_COMPONENTS, pieces, strict=True):
                    counts[name] += _scalar_json_bytes(piece) - 2
            else:
                counts[_CONTENT_COMPONENT[role]] += _scalar_json_bytes(content) - 2
            for key, value in message.items():
                if key == "content":
                    continue
                component = "other_metadata"
                if role == "assistant" and key == "tool_calls":
                    component = "assistant"
                elif role == "tool" and key == "tool_call_id":
                    component = "tool_result"
                counts[component] += _compact_json_size({key: value}, self._max_bytes) - 2

        for key, value in body.items():
            if key != "messages":
                component = "tool_definition" if key == "tools" else "other_metadata"
                counts[component] += _compact_json_size({key: value}, self._max_bytes) - 2
        return counts
