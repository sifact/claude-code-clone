"""bash - runs a shell command inside the sandbox, with a timeout."""

import os
import subprocess
from typing import Any

from .sandbox import DEFAULT_TIMEOUT, MAX_OUTPUT, WORKDIR, truncate


def bash(tool_input: dict[str, Any]) -> dict[str, Any]:
    timeout = tool_input.get("timeout", DEFAULT_TIMEOUT)
    try:
        proc = subprocess.run(
            ["bash", "-c", tool_input["command"]],
            cwd=WORKDIR,
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
