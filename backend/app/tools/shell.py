"""bash - runs a shell command inside the sandbox, with a timeout."""

import os
import subprocess
from pathlib import Path
from typing import Any

from .sandbox import DEFAULT_TIMEOUT, MAX_OUTPUT, truncate


def bash(tool_input: dict[str, Any], workdir: Path) -> dict[str, Any]:
    timeout = tool_input.get("timeout", DEFAULT_TIMEOUT)
    try:
        proc = subprocess.run(
            ["bash", "-c", tool_input["command"]],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "TERM": "dumb"},
        )
        return {
            "stdout": truncate(proc.stdout, MAX_OUTPUT),
            "stderr": truncate(proc.stderr, MAX_OUTPUT),
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        return {"stdout": truncate(exc.stdout or "", MAX_OUTPUT), "stderr": "Command timed out", "exit_code": -1}
