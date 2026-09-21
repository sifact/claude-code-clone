"""FastAPI entrypoint. Two endpoints: POST /chat starts a turn, POST
/chat/respond continues one paused on a human approval decision.

No SESSIONS or PENDING_APPROVALS dict here, unlike the hand-rolled version -
the LangGraph checkpointer owns all of that now, keyed by `session_id` as
its `thread_id`. That's the whole comparison this branch exists to show.
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel

load_dotenv()

from .agent import RECURSION_LIMIT, agent_graph, initial_input  # noqa: E402
from .tools import DEFAULT_WORKDIR  # noqa: E402

app = FastAPI(title="nightcode-fastapi")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "BUILD"
    # Sent by the CLI as its process.cwd(), so tools operate on whatever
    # project you actually launched it from. The browser client has no real
    # filesystem to point at, so it omits this and falls back to
    # DEFAULT_WORKDIR (a fixed folder inside the backend).
    cwd: str | None = None


class RespondRequest(BaseModel):
    session_id: str
    decision: Literal["approve", "deny"]


def resolve_workdir(cwd: str | None) -> Path:
    if cwd is None:
        return DEFAULT_WORKDIR

    workdir = Path(cwd).resolve()
    if not workdir.is_dir():
        raise HTTPException(400, f"cwd is not a directory: {workdir}")
    return workdir


def has_pending_approval(config: dict[str, Any]) -> bool:
    # A non-empty `.next` means the graph stopped mid-run - the only way
    # that happens here is an unresolved interrupt() in run_one_tool.
    return bool(agent_graph.get_state(config).next)


def sse_events(step_input: Any, config: dict[str, Any]) -> Generator[str, None, None]:
    """Drives the graph and formats its "custom" events (our own event
    vocabulary, emitted via get_stream_writer() in graph.py) as SSE lines.
    Interrupts don't come through that channel - they're a distinct
    LangGraph control-flow signal - so they're translated here instead.

    Both frontends expect a final `done` event to clear their streaming
    cursor - the graph itself doesn't emit one (it just stops), so this
    adds it explicitly, except when the turn paused on an approval instead
    (that already clears the cursor client-side on `approval_required`).
    """
    interrupted = False
    try:
        for mode, chunk in agent_graph.stream(
            step_input, config, stream_mode=["custom", "updates"], recursion_limit=RECURSION_LIMIT
        ):
            if mode == "custom":
                yield f"data: {json.dumps(chunk)}\n\n"
            elif mode == "updates" and "__interrupt__" in chunk:
                interrupted = True
                approval = {"type": "approval_required", **chunk["__interrupt__"][0].value}
                yield f"data: {json.dumps(approval)}\n\n"
        if not interrupted:
            yield f"data: {json.dumps({'type': 'done', 'stop_reason': 'stop'})}\n\n"
    except GraphRecursionError:
        yield f"data: {json.dumps({'type': 'done', 'stop_reason': 'max_steps_exceeded'})}\n\n"
    except Exception as error:  # noqa: BLE001 - last line of defense so the
        # SSE stream always terminates with an event the client can render,
        # instead of dying mid-response and leaving the frontend hanging.
        # (call_model already emits a `error` custom event before raising,
        # for a clean model/API failure - this also catches anything else.)
        print(f"Unhandled error in agent graph: {error}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(error)})}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    if has_pending_approval(config):
        raise HTTPException(409, "Resolve the pending tool approval (/chat/respond) before sending a new message")

    workdir = resolve_workdir(req.cwd)
    step_input = initial_input(req.message, req.mode, workdir)

    return StreamingResponse(sse_events(step_input, config), media_type="text/event-stream")


@app.post("/chat/respond")
def respond(req: RespondRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    if not has_pending_approval(config):
        raise HTTPException(404, "No pending approval for this session")

    return StreamingResponse(sse_events(Command(resume=req.decision), config), media_type="text/event-stream")


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    return {
        "role": message.type,
        "content": message.content,
        **({"tool_calls": message.tool_calls} if getattr(message, "tool_calls", None) else {}),
        **({"tool_call_id": message.tool_call_id} if hasattr(message, "tool_call_id") else {}),
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    messages = agent_graph.get_state(config).values.get("messages", [])
    return {"messages": [_serialize_message(m) for m in messages]}
