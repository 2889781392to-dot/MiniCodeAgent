"""Settings for the synchronous Chat Completions client and agent limits."""

from collections.abc import Mapping
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator

from .protocol import ProtocolModel


API_KEY_ENV = "MINICODEAGENT_API_KEY"
_BASE_URL_ENV = "MINICODEAGENT_BASE_URL"
_MODEL_ENV = "MINICODEAGENT_MODEL"
_TIMEOUT_ENV = "MINICODEAGENT_TIMEOUT"
_THINKING_ENV = "MINICODEAGENT_DEEPSEEK_THINKING"
_BASE_URL_ERROR = f"{_BASE_URL_ENV} must be an HTTP(S) URL without credentials."
_TIMEOUT_ERROR = f"{_TIMEOUT_ENV} must be a positive finite number of seconds."


class LoopLimits(ProtocolModel):
    max_provider_calls: int = Field(default=12, ge=1)
    max_tool_batches: int = Field(default=8, ge=0)
    max_calls_per_batch: int = Field(default=8, ge=1)


class Settings(ProtocolModel):
    api_key: SecretStr
    model_name: str
    base_url: str | None = None
    request_timeout: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    deepseek_thinking: Literal["disabled"] | None = None
    limits: LoopLimits = Field(default_factory=LoopLimits)

    @field_validator("base_url")
    @classmethod
    def check_base_url(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                url = urlsplit(value)
                if url.scheme not in {"http", "https"} or not url.hostname:
                    raise ValueError
                if url.username is not None or url.password is not None:
                    raise ValueError
                _ = url.port
            except ValueError:
                raise ValueError(_BASE_URL_ERROR) from None
        return value

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "Settings":
        for name in (API_KEY_ENV, _MODEL_ENV):
            if not environ.get(name, "").strip():
                raise ValueError(f"Set {name} before running MiniCodeAgent.")

        thinking = environ.get(_THINKING_ENV) or None
        if thinking is not None and thinking != "disabled":
            raise ValueError(
                f"{_THINKING_ENV} only supports 'disabled' in V0.1. "
                "Leave it unset for a generic OpenAI-compatible endpoint."
            )

        try:
            timeout = float(environ.get(_TIMEOUT_ENV, "60"))
        except ValueError:
            raise ValueError(_TIMEOUT_ERROR) from None
        if not 0 < timeout < float("inf"):
            raise ValueError(_TIMEOUT_ERROR)

        try:
            return cls(
                api_key=SecretStr(environ[API_KEY_ENV]),
                model_name=environ[_MODEL_ENV],
                base_url=environ.get(_BASE_URL_ENV) or None,
                request_timeout=timeout,
                deepseek_thinking=thinking,
            )
        except ValueError:
            # A URL validation error can contain the input, including accidental credentials.
            raise ValueError(_BASE_URL_ERROR) from None
