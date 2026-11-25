"""Run the CLI without inheriting another application's OpenAI SDK settings."""

import os
import subprocess
import sys

from .environment import is_openai_variable, isolated_environment


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if any(is_openai_variable(name) for name in os.environ):
        # The SDK itself reads ambient settings at import and construction.
        # Start it with a filtered child environment and leave the parent alone.
        try:
            return subprocess.call(
                [sys.executable, "-m", "minicodeagent", *args],
                env=isolated_environment(os.environ),
            )
        except KeyboardInterrupt:
            return 130

    from .cli import main as run_cli

    return run_cli(args)

if __name__ == "__main__":
    raise SystemExit(main())
