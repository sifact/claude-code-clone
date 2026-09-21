"""The hand-rolled agent loop: no LangChain, no LangGraph, no AI SDK.

Ask the model -> stream text and tool-call fragments -> run any tool calls ->
loop, until the model stops asking for tools or MAX_STEPS is hit. No agent
object, no graph, no planner - just a while loop over a growing message list.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from typing import Any

from openai import APIConnectionError, APIError

from ..system_prompt import build_system_prompt
from ..tools import ToolError, execute_tool, get_tool_schemas
from .dedup import IDEMPOTENT_READ_TOOLS, already_called
from .provider import MAX_STEPS, MAX_TOKENS, MAX_VALIDATION_RETRIES, MODEL, client


def run_agent_loop(
    history: list[dict[str, Any]],
    mode: str,
) -> Generator[dict[str, Any], None, None]:
    """Yields SSE-ready event dicts. Mutates `history` in place as the
    session's message list grows across turns.
    """

    messages = history
    tools = get_tool_schemas(mode)

    system_message = {"role": "system", "content": build_system_prompt(mode)}
    if messages and messages[0].get("role") == "system":
        messages[0] = system_message  # keep it current if mode changed since the last turn
    else:
        messages.insert(0, system_message)

    for _step in range(MAX_STEPS):
        print(f"[agent] step {_step + 1}/{MAX_STEPS} starting, {len(messages)} messages in history", flush=True)
        for attempt in range(MAX_VALIDATION_RETRIES + 1):
            assistant_text = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            step_start = time.monotonic()

            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    tools=tools,
                    stream=True,
                )
                print(f"[agent] step {_step + 1} attempt {attempt + 1}: request sent, awaiting first chunk", flush=True)

                for chunk in stream:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    # delta.content is the final answer; some models also
                    # stream a separate reasoning field, intentionally
                    # ignored here since it isn't the response.
                    if delta.content:
                        assistant_text += delta.content
                        yield {"type": "text_delta", "text": delta.content}

                    for tc_delta in delta.tool_calls or []:
                        # A tool call's id/name arrive once, then its
                        # arguments trickle in as partial JSON-string chunks
                        # keyed by index, concatenated once the stream ends.
                        entry = tool_calls_acc.setdefault(tc_delta.index, {"id": None, "name": None, "arguments": ""})
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

                print(
                    f"[agent] step {_step + 1} attempt {attempt + 1}: stream finished after "
                    f"{time.monotonic() - step_start:.1f}s, {len(tool_calls_acc)} tool call(s)",
                    flush=True,
                )
                break  # streamed successfully, move on to handling the result
            except APIError as error:
                print(
                    f"[agent] step {_step + 1} attempt {attempt + 1}: {type(error).__name__} after "
                    f"{time.monotonic() - step_start:.1f}s: {error}",
                    flush=True,
                )
                if attempt >= MAX_VALIDATION_RETRIES:
                    yield {"type": "error", "message": str(error)}
                    return

                # Connection/timeout errors aren't the model's fault - retry
                # silently. A rejected generation (e.g. a tool call missing a
                # required argument) gets a corrective message instead, since
                # the model never otherwise learns that attempt failed.
                if not isinstance(error, APIConnectionError):
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Your last tool call attempt was rejected: {error}. "
                            "Try again, making sure to include every required parameter.",
                        }
                    )
                yield {"type": "retry", "attempt": attempt + 1, "reason": str(error)}

        ordered_tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]

        assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_text or None}
        if ordered_tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in ordered_tool_calls
            ]
        messages.append(assistant_message)

        if not ordered_tool_calls:
            yield {"type": "done", "stop_reason": "stop"}
            return

        # Snapshot before this step's calls execute, so a duplicate within
        # the same step can't spuriously match against itself.
        prior_messages = messages[:-1]

        for tc in ordered_tool_calls:
            try:
                tool_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": "error: malformed tool call arguments"}
                )
                yield {"type": "tool_error", "tool_call_id": tc["id"], "error": "malformed tool call arguments"}
                continue

            yield {"type": "tool_call", "tool_call_id": tc["id"], "name": tc["name"], "input": tool_input}

            if tc["name"] in IDEMPOTENT_READ_TOOLS and already_called(prior_messages, tc["name"], tool_input):
                note = {
                    "note": f"You already called {tc['name']} with these exact arguments earlier in this "
                    "conversation. Re-use that earlier result instead of repeating the call."
                }
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(note)})
                yield {"type": "tool_skipped", "tool_call_id": tc["id"], "name": tc["name"]}
                continue

            try:
                output = execute_tool(tc["name"], tool_input, mode)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(output)})
                yield {"type": "tool_result", "tool_call_id": tc["id"], "output": output}
            except ToolError as error:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": f"error: {error}"})
                yield {"type": "tool_error", "tool_call_id": tc["id"], "error": str(error)}

    yield {"type": "done", "stop_reason": "max_steps_exceeded"}
