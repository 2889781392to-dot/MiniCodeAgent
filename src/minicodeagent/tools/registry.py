"""Bind public tool specifications to validated local handlers."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ValidationError

from ..protocol import ToolCall, ToolResult, ToolSpec, parse_tool_arguments
from ..workspace import Workspace, WorkspaceError

MAX_OUTPUT_CHARS = 20_000
MAX_DETAILS_BYTES = 128 * 1024


class ToolFailure(ValueError):
    """A controlled tool diagnostic that can be returned to the model."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, Workspace, UUID], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolSpec]] = {}
        self._in_use = False

    def register(self, definition: ToolDefinition) -> None:
        if self._in_use:
            raise RuntimeError("Tool bindings are in use by a model request.")
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        spec = ToolSpec(
            name=definition.name,
            description=definition.description,
            parameters=definition.args_model.model_json_schema(),
        )
        self._tools[definition.name] = (definition, spec)

    def definitions(self) -> tuple[ToolSpec, ...]:
        return tuple(spec.model_copy(deep=True) for _, spec in self._tools.values())

    @contextmanager
    def fixed_bindings(self) -> Iterator[None]:
        if self._in_use:
            raise RuntimeError("Tool bindings are already in use.")
        self._in_use = True
        try:
            yield
        finally:
            self._in_use = False

    def execute(self, call: ToolCall, workspace: Workspace) -> ToolResult:
        binding = self._tools.get(call.name)
        if binding is None:
            return self._error(call, "unknown_tool", f"Unknown tool: {call.name[:100]}")

        tool, spec = binding
        try:
            parsed = parse_tool_arguments(call.arguments_raw)
            # Tool arguments are objects. Reject unknown names even if a supplied
            # Pydantic model accepts extra fields by default.
            properties = spec.parameters.get("properties", {})
            if isinstance(properties, dict) and parsed.keys() - properties.keys():
                return self._error(call, "invalid_arguments", "Unexpected tool argument fields.")
            arguments = tool.args_model.model_validate(parsed, strict=True)
        except ValidationError as exc:
            # Input values are omitted. Field names and messages suffice for repair.
            problems = []
            for error in exc.errors(include_input=False, include_url=False)[:5]:
                field = ".".join(map(str, error["loc"]))[:100] or "arguments"
                problems.append(f"{field}: {error['msg']}")
            return self._error(call, "invalid_arguments", "Invalid arguments: " + "; ".join(problems))
        except (ValueError, RecursionError, OverflowError):
            return self._error(call, "invalid_arguments", "Arguments must be a valid JSON object.")

        try:
            result = tool.handler(arguments, workspace, call.call_id)
            if not isinstance(result, ToolResult) or result.call_id != call.call_id:
                raise ToolFailure("Tool returned an invalid result or a mismatched call ID.")
            result = ToolResult.model_validate(result.model_dump(mode="python", warnings=False), strict=True)
            if len(result.output) > MAX_OUTPUT_CHARS:
                marker = "\n[tool output truncated]\n"
                keep = (MAX_OUTPUT_CHARS - len(marker)) // 2
                result = result.model_copy(update={
                    "output": result.output[:keep] + marker + result.output[-keep:],
                    "details": {**result.details, "output_truncated": True},
                })
            if len(json.dumps(result.details, allow_nan=False)) > MAX_DETAILS_BYTES:
                raise ToolFailure("Tool returned oversized result details.")
        except (WorkspaceError, ToolFailure) as exc:
            return self._error(call, "tool_error", str(exc))
        except Exception:
            return self._error(call, "tool_error", "Tool execution failed or returned invalid result details.")

        return result

    @staticmethod
    def _error(call: ToolCall, code: str, message: str) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            status="error",
            output=message[:MAX_OUTPUT_CHARS],
            details={"error_code": code},
        )
