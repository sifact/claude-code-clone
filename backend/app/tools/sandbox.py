"""Shared sandbox helpers: path resolution, size limits, ToolError."""

import os
from pathlib import Path

WORKDIR = Path(os.environ.get("AGENT_WORKDIR", Path(__file__).parent.parent.parent / "workspace")).resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10_000
MAX_RESULTS = 200
MAX_MATCHES = 50
MAX_OUTPUT = 20_000
DEFAULT_TIMEOUT = 30


class ToolError(Exception):
    pass


def resolve_inside_workdir(path: str) -> Path:
    resolved = (WORKDIR / path).resolve()
    try:
        resolved.relative_to(WORKDIR)
    except ValueError:
        raise ToolError("Path is outside the sandboxed working directory")
    return resolved


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n... (truncated, {len(value)} total chars)"
