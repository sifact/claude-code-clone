"""Detects a repeated read-only tool call so the loop can skip re-running it.

Models sometimes ignore the system prompt's "don't re-read files" rule and
burn their step budget re-exploring instead of acting. Enforced here since
the model can't be relied on to police it itself.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

IDEMPOTENT_READ_TOOLS = {"read_file", "list_directory", "glob", "grep"}
MUTATING_TOOLS = {"write_file", "edit_file", "bash"}


def already_called(prior_messages: list[BaseMessage], name: str, tool_input: dict[str, Any]) -> bool:
    """True if this exact idempotent call already ran, with no write/edit/bash since."""
    seen: set[tuple[str, str]] = set()
    for message in prior_messages:
        if not isinstance(message, AIMessage):
            continue
        for tc in message.tool_calls:
            if tc["name"] in MUTATING_TOOLS:
                seen.clear()  # filesystem may have changed; forget prior reads
                continue
            if tc["name"] in IDEMPOTENT_READ_TOOLS:
                seen.add((tc["name"], json.dumps(tc["args"], sort_keys=True)))
    return (name, json.dumps(tool_input, sort_keys=True)) in seen
