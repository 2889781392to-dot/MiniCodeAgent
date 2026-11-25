"""Build child environments without fetching excluded variable values."""

from collections.abc import Mapping


def is_openai_variable(name: str) -> bool:
    return name.upper().startswith("OPENAI_")


def isolated_environment(environ: Mapping[str, str], *, omit_key: str | None = None) -> dict[str, str]:
    return {
        name: environ[name]
        for name in environ
        if not is_openai_variable(name) and name.upper() != omit_key
    }
