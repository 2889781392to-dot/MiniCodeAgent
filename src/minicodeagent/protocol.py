"""Internal types shared by the agent, tools, context, and model adapter."""

import json
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


NonBlank = Annotated[str, Field(min_length=1, pattern=r"\S")]
ToolStatus = Literal["ok", "error", "timeout", "interrupted"]
TurnStatus = Literal["completed", "limited", "failed", "interrupted"]
ModelErrorKind = Literal[
    "authentication", "timeout", "rate_limit", "server_error", "connection",
    "context_length", "compatibility_error", "api_error", "protocol_error",
]


class ToolSpec(ProtocolModel):
    name: NonBlank
    description: str
    parameters: dict[str, JsonValue]


class ToolCall(ProtocolModel):
    provider_call_id: NonBlank
    name: NonBlank
    arguments_raw: str
    call_id: UUID = Field(default_factory=uuid4)

    @property
    def wire_id(self) -> str:
        return wire_call_id(self.call_id)


def wire_call_id(call_id: UUID) -> str:
    return f"call_{call_id.hex}"


def parse_tool_arguments(raw: str) -> dict[str, JsonValue]:
    arguments = json.loads(raw)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments are not an object.")
    json.dumps(arguments, allow_nan=False)
    return arguments


class ToolResult(ProtocolModel):
    call_id: UUID
    status: ToolStatus
    output: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ChatMessage(ProtocolModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    call_id: UUID | None = None


class ChatRequest(ProtocolModel):
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...] = ()


class TokenUsage(ProtocolModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class PromptTokenDetails(ProtocolModel):
    cached_tokens: int | None = Field(default=None, ge=0)
    audio_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)


class ModelReply(ProtocolModel):
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: NonBlank
    usage: TokenUsage | None = None
    prompt_tokens_details: PromptTokenDetails | None = None


class ModelError(Exception):
    """A classified model failure with a diagnostic safe for the CLI."""

    def __init__(self, kind: ModelErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind

    @property
    def retryable(self) -> bool:
        return self.kind in {"timeout", "rate_limit", "server_error", "connection"}


class TurnOutcome(ProtocolModel):
    turn_id: UUID
    status: TurnStatus
    reason: str
    content: str = ""
    assistant_seq: int | None = None
