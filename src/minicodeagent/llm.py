"""Translate internal messages to synchronous OpenAI Chat Completions."""

from collections.abc import Callable

from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from .chat_payload import encode_chat_payload
from .config import Settings
from .protocol import (
    ChatRequest,
    ModelError,
    ModelReply,
    PromptTokenDetails,
    TokenUsage,
    ToolCall,
)


class LLMClient:
    def __init__(self, settings: Settings, *, sdk_client: OpenAI | None = None) -> None:
        self._settings = settings
        self._sdk = sdk_client if sdk_client is not None else OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url or "https://api.openai.com/v1",
            timeout=settings.request_timeout,
            max_retries=0,
        )

    def close(self) -> None:
        self._sdk.close()

    def complete_with_payload(self, request: ChatRequest, observe: Callable[[dict], None]) -> ModelReply:
        """Observe the actual prepared payload immediately before calling the SDK."""
        return self.complete(request, observe=observe)

    def request_payload(self, request: ChatRequest) -> dict:
        return encode_chat_payload(
            request, model=self._settings.model_name, deepseek_thinking=self._settings.deepseek_thinking,
        )

    def summary_payload(self, request: ChatRequest, *, max_output_tokens: int) -> dict:
        """Prepare the exact bounded, tool-free payload for a handoff request."""
        if (
            request.tools or tuple(message.role for message in request.messages) != ("system", "user")
            or any(message.tool_calls or message.call_id is not None or not message.content.strip()
                   for message in request.messages)
        ):
            raise ModelError("protocol_error", "A summary request requires only a system prompt and user data.")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ModelError("protocol_error", "A summary request requires a positive output token limit.")
        return {**self.request_payload(request), "tool_choice": "none", "max_tokens": max_output_tokens}

    def summarize(self, request: ChatRequest, *, max_output_tokens: int) -> ModelReply:
        """Call the configured model once for a text-only handoff."""
        params = self.summary_payload(request, max_output_tokens=max_output_tokens)
        reply = self._complete_payload(params)
        if reply.finish_reason != "stop" or reply.tool_calls or not reply.content.strip():
            raise ModelError("protocol_error", "The endpoint did not return a complete, text-only handoff.")
        return reply

    def complete(self, request: ChatRequest, *, observe: Callable[[dict], None] | None = None) -> ModelReply:
        return self._complete_payload(self.request_payload(request), observe=observe)

    def _complete_payload(self, params: dict, *, observe: Callable[[dict], None] | None = None) -> ModelReply:
        try:
            create = self._sdk.chat.completions.create
            if observe is not None:
                observe(params)
            response = create(**params)
        except APITimeoutError:
            raise ModelError("timeout", "The model request timed out.") from None
        except APIConnectionError:
            raise ModelError("connection", "Could not connect to the model endpoint.") from None
        except APIStatusError as exc:
            raise self._status_error(exc) from None
        except APIResponseValidationError:
            raise ModelError("protocol_error", "The endpoint returned an invalid chat response.") from None
        except APIError:
            raise ModelError("api_error", "The model request failed.") from None
        except Exception:
            raise ModelError("api_error", "The model client could not send the request.") from None

        try:
            choice = response.choices[0]
            message = choice.message
            if getattr(message, "reasoning_content", None):
                raise ModelError(
                    "compatibility_error",
                    "The endpoint returned reasoning_content. V0.1 requires non-thinking tool calls. "
                    "Check that this endpoint supports DeepSeek thinking.type=disabled.",
                )
            finish_reason = choice.finish_reason
            content = message.content if message.content is not None else ""
            calls = []
            if finish_reason not in {"length", "content_filter"}:
                for item in message.tool_calls or ():
                    if item.type != "function":
                        raise ValueError("Unsupported tool type.")
                    calls.append(ToolCall(
                        provider_call_id=item.id, name=item.function.name, arguments_raw=item.function.arguments,
                    ))
            usage = None
            details = None
            sdk_usage = getattr(response, "usage", None)
            if sdk_usage is not None:
                usage = TokenUsage(
                    input_tokens=sdk_usage.prompt_tokens,
                    output_tokens=sdk_usage.completion_tokens,
                    total_tokens=sdk_usage.total_tokens,
                )
                raw_details = getattr(sdk_usage, "prompt_tokens_details", None)
                known = {}
                for key in ("cached_tokens", "audio_tokens", "cache_write_tokens"):
                    value = getattr(raw_details, key, None)
                    if type(value) is int and value >= 0:
                        known[key] = value
                if known:
                    details = PromptTokenDetails(**known)
            return ModelReply(
                content=content, tool_calls=tuple(calls), finish_reason=finish_reason, usage=usage,
                prompt_tokens_details=details,
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            raise ModelError("protocol_error", "The endpoint returned a malformed chat response.") from None

    def _status_error(self, error: APIStatusError) -> ModelError:
        status = error.status_code
        body = error.body if isinstance(error.body, dict) else {}
        if isinstance(body.get("error"), dict):
            body = body["error"]
        code = str(body.get("code") or "").casefold()
        # This text is used only to classify the response and is never displayed.
        diagnostic = " ".join(str(body.get(key) or "") for key in ("param", "code", "message"))
        diagnostic = (diagnostic + " " + error.message).casefold()
        if status in {401, 403}:
            return ModelError("authentication", f"Model endpoint denied access (HTTP {status}).")
        if status in {400, 422}:
            if self._settings.deepseek_thinking == "disabled" and "thinking" in diagnostic:
                return ModelError(
                    "compatibility_error",
                    f"Endpoint rejected DeepSeek thinking.type=disabled (HTTP {status}). "
                    "Check this channel's support for the parameter.",
                )
            if code == "context_length_exceeded" or "maximum context length" in diagnostic:
                return ModelError("context_length", "The model endpoint rejected the context length.")
        if status == 408:
            return ModelError("timeout", "The model endpoint reported a timeout (HTTP 408).")
        if status == 429:
            return ModelError("rate_limit", "The model endpoint is rate limiting requests (HTTP 429).")
        if 500 <= status <= 599:
            return ModelError("server_error", f"The model endpoint failed (HTTP {status}).")
        return ModelError("api_error", f"The model endpoint rejected the request (HTTP {status}).")

