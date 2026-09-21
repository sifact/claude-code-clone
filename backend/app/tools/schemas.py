"""Tool schemas offered to the model. `description`/`parameters` below are
read by the LLM at inference time to decide when and how to call each tool -
edit them for accuracy, not just readability.
"""

from typing import Any

READ_ONLY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file's contents from the sandboxed working directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path to the file"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List entries in a directory under the sandboxed working directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative directory path", "default": "."}},
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern (e.g. '**/*.py').",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files"},
                "path": {"type": "string", "description": "Directory to search from", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search file contents with a regular expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search from", "default": "."},
                "include": {"type": "string", "description": "Optional glob for files to include"},
            },
            "required": ["pattern"],
        },
    },
]

BUILD_ONLY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "write_file",
        "description": "Create or overwrite a file under the sandboxed working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to write"},
                "content": {"type": "string", "description": "File contents"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file. old_string must be unique in the file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to edit"},
                "old_string": {"type": "string", "description": "Exact text to replace; must be unique"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command inside the sandboxed working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "number", "description": "Timeout in seconds"},
            },
            "required": ["command"],
        },
    },
]

BUILD_TOOLS = READ_ONLY_TOOLS + BUILD_ONLY_TOOLS
READ_ONLY_TOOL_NAMES = {t["name"] for t in READ_ONLY_TOOLS}


def _as_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }, 
    }


def get_tool_schemas(mode: str) -> list[dict[str, Any]]:
    catalog = READ_ONLY_TOOLS if mode == "PLAN" else BUILD_TOOLS
    return [_as_openai_tool(t) for t in catalog]
