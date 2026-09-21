"""The same agent loop as the hand-rolled version, rebuilt as a LangGraph
graph - this is the direct point of comparison with agent/loop.py.

Everything that file did by hand, this gets from the framework:
  - state persistence across turns (SESSIONS dict there -> checkpointer here,
    keyed by `thread_id`)
  - pausing for a human decision (PENDING_APPROVALS + a custom /chat/respond
    resume protocol there -> `interrupt()` + `Command(resume=...)` here)
  - a step budget (MAX_STEPS there -> `recursion_limit` here)

One tool call per node invocation, not a Python loop over all of them. This
is deliberate, not just a style choice: LangGraph re-runs a node from the
top when resuming after `interrupt()`, so any side effect in that node
*before* the interrupt call would silently re-execute too. Keeping "run one
tool" as its own node - looped via a graph edge, not a for-loop - means each
invocation calls interrupt() at most once, before any side effect, so
there's nothing earlier in that same call that resuming could replay.
"""

import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import APIConnectionError, APIError

from ..system_prompt import build_system_prompt
from ..tools import ToolError, execute_tool, get_tool_schemas
from .approval import APPROVAL_REQUIRED_TOOLS
from .dedup import IDEMPOTENT_READ_TOOLS, already_called
from .provider import MAX_VALIDATION_RETRIES, llm
from .state import AgentState


def _next_unanswered_tool_call(state: AgentState) -> tuple[AIMessage, dict] | tuple[None, None]:
    """The most recent AIMessage and the next tool call on it that doesn't
    have a matching ToolMessage yet, or (None, None) if there isn't one.
    """
    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if not last_ai or not last_ai.tool_calls:
        return None, None
    answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
    tc = next((tc for tc in last_ai.tool_calls if tc["id"] not in answered), None)
    return (last_ai, tc) if tc else (None, None)


def call_model(state: AgentState) -> dict:
    writer = get_stream_writer()
    tools = get_tool_schemas(state["mode"])
    llm_with_tools = llm.bind_tools(tools)
    messages = state["messages"]

    for attempt in range(MAX_VALIDATION_RETRIES + 1):
        try:
            chunks = []
            for chunk in llm_with_tools.stream(messages):
                if chunk.content:
                    writer({"type": "text_delta", "text": chunk.content})
                chunks.append(chunk)
            response = chunks[0]
            for chunk in chunks[1:]:
                response = response + chunk
            return {"messages": [response]}
        except APIError as error:
            if attempt >= MAX_VALIDATION_RETRIES:
                writer({"type": "error", "message": str(error)})
                raise
            if isinstance(error, APIConnectionError):
                # Not the model's fault - retry the exact same request.
                writer({"type": "retry", "attempt": attempt + 1, "reason": str(error)})
                continue
            # The API rejected the model's own generation (e.g. a tool call
            # missing a required argument). It never otherwise learns that
            # failed, so inject a corrective message before retrying.
            messages = [
                *messages,
                HumanMessage(
                    content=f"Your last tool call attempt was rejected: {error}. "
                    "Try again, making sure to include every required parameter."
                ),
            ]
            writer({"type": "retry", "attempt": attempt + 1, "reason": str(error)})

    raise RuntimeError("unreachable: loop always returns or raises")


def run_one_tool(state: AgentState) -> dict:
    writer = get_stream_writer()
    triggering_message, tc = _next_unanswered_tool_call(state)
    assert tc is not None, "run_one_tool entered with nothing to do - routing bug"

    if tc["name"] == "ask_question":
        # A different reason to interrupt() than APPROVAL_REQUIRED_TOOLS
        # below - not "may I run this," but "what should I do." The resume
        # value is the option text the user picked, not approve/deny, and
        # it becomes the tool's result directly - there's no real tool
        # execution here, `execute_tool` never sees this tool name.
        answer = interrupt(
            {
                "kind": "clarification",
                "tool_call_id": tc["id"],
                "question": tc["args"]["question"],
                "options": tc["args"]["options"],
            }
        )
        return {"messages": [ToolMessage(content=answer, tool_call_id=tc["id"])]}

    mode = state["mode"]
    workdir = Path(state["workdir"])
    # Exclude the AIMessage that requested this call from the dedup check -
    # otherwise this call always "matches itself". Identity, not equality:
    # two calls can be textually identical without being the same object.
    prior_messages = [m for m in state["messages"] if m is not triggering_message]

    if tc["name"] in APPROVAL_REQUIRED_TOOLS:
        # Raises GraphInterrupt the first time through; the checkpointer
        # persists everything and the graph stops here until a
        # Command(resume=...) arrives. On resume, this node re-runs from the
        # top - nothing above this line has a side effect, so that's safe.
        decision = interrupt({"kind": "approval", "tool_call_id": tc["id"], "name": tc["name"], "input": tc["args"]})
        if decision == "deny":
            writer({"type": "tool_denied", "tool_call_id": tc["id"]})
            return {
                "messages": [
                    ToolMessage(content="User denied this action; it was not run.", tool_call_id=tc["id"])
                ]
            }

    writer({"type": "tool_call", "tool_call_id": tc["id"], "name": tc["name"], "input": tc["args"]})

    if tc["name"] in IDEMPOTENT_READ_TOOLS and already_called(prior_messages, tc["name"], tc["args"]):
        note = {
            "note": f"You already called {tc['name']} with these exact arguments earlier in this "
            "conversation. Re-use that earlier result instead of repeating the call."
        }
        writer({"type": "tool_skipped", "tool_call_id": tc["id"], "name": tc["name"]})
        return {"messages": [ToolMessage(content=json.dumps(note), tool_call_id=tc["id"])]}

    try:
        output = execute_tool(tc["name"], tc["args"], mode, workdir)
        writer({"type": "tool_result", "tool_call_id": tc["id"], "output": output})
        return {"messages": [ToolMessage(content=json.dumps(output), tool_call_id=tc["id"])]}
    except ToolError as error:
        writer({"type": "tool_error", "tool_call_id": tc["id"], "error": str(error)})
        return {"messages": [ToolMessage(content=f"error: {error}", tool_call_id=tc["id"])]}


def route_after_model(state: AgentState) -> Literal["run_one_tool", "__end__"]:
    _, tc = _next_unanswered_tool_call(state)
    return "run_one_tool" if tc else END


def route_after_tool(state: AgentState) -> Literal["run_one_tool", "call_model"]:
    _, tc = _next_unanswered_tool_call(state)
    return "run_one_tool" if tc else "call_model"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("run_one_tool", run_one_tool)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", route_after_model)
    graph.add_conditional_edges("run_one_tool", route_after_tool)

    # In-memory: same "no database" simplification as the hand-rolled
    # version's SESSIONS dict - resets if the server restarts.
    return graph.compile(checkpointer=InMemorySaver())


agent_graph = build_graph()


def initial_input(message: str, mode: str, workdir: Path) -> AgentState:
    return {
        "messages": [
            # A stable id means add_messages *replaces* this on the next
            # turn instead of appending a second system message - same
            # "keep it current if mode changed" behavior as the hand-rolled
            # version's `messages[0] = system_message`.
            SystemMessage(content=build_system_prompt(mode), id="system"),
            HumanMessage(content=message),
        ],
        "mode": mode,
        "workdir": str(workdir),
    }
