"""Shared sandbox helpers: path resolution, size limits, ToolError.

Every tool is sandboxed to a `workdir` passed in per-request, not a fixed
global - the browser client gets DEFAULT_WORKDIR (a fixed folder, since a
browser has no real filesystem to point at), while the CLI sends its own
process.cwd() so it can operate on whatever project you actually launched it
from, the same way nightcode's real CLI and Claude Code do.
"""

import os
from pathlib import Path

DEFAULT_WORKDIR = Path(os.environ.get("AGENT_WORKDIR", Path(__file__).parent.parent.parent / "workspace")).resolve()
DEFAULT_WORKDIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10_000
MAX_RESULTS = 200
MAX_MATCHES = 50
MAX_OUTPUT = 20_000
DEFAULT_TIMEOUT = 30


class ToolError(Exception):
    pass


def resolve_inside_workdir(path: str, workdir: Path) -> Path:
    resolved = (workdir / path).resolve()
    try:
        resolved.relative_to(workdir)
    except ValueError:
        raise ToolError("Path is outside the sandboxed working directory")
    return resolved


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n... (truncated, {len(value)} total chars)"
