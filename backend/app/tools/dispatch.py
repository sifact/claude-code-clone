"""Routes a tool call by name to its handler function."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .filesystem import edit_file, glob_files, grep_files, list_directory, read_file, write_file
from .sandbox import ToolError
from .schemas import READ_ONLY_TOOL_NAMES
from .shell import bash

TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Path], dict[str, Any]]] = {
    "read_file": read_file,
    "list_directory": list_directory,
    "glob": glob_files,
    "grep": grep_files,
    "write_file": write_file,
    "edit_file": edit_file,
    "bash": bash,
}


def execute_tool(name: str, tool_input: dict[str, Any], mode: str, workdir: Path) -> dict[str, Any]:
    if mode == "PLAN" and name not in READ_ONLY_TOOL_NAMES:
        raise ToolError(f"Tool {name} is not available in PLAN mode")

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ToolError(f"Unknown tool: {name}")

    try:
        return handler(tool_input, workdir)
    except ToolError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        # Anything the filesystem itself throws that isn't already a
        # ToolError (missing file, bad encoding, etc) - without this, one
        # bad path crashes the whole SSE stream instead of surfacing as a
        # normal tool_error.
        raise ToolError(str(error)) from error
