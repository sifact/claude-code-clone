"""FastAPI entrypoint. Two endpoints: POST /chat starts a turn, POST
/chat/respond continues one that paused for a human approval decision.

Sessions are an in-memory dict (list of OpenAI-format chat messages per
session id) - no database. That's a deliberate simplification to keep this
phase focused on the agent loop itself, not persistence.
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from .agent import run_agent_loop  # noqa: E402
from .tools import DEFAULT_WORKDIR, ToolError, execute_tool  # noqa: E402

app = FastAPI(title="nightcode-fastapi")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, list[dict]] = {}
# session_id -> {"tool_calls": [...], "step": int, "prior_len": int, "mode": str, "workdir": str}
# Present only while a step is paused waiting on a write_file/edit_file/bash
# approval - see agent/loop.py's run_tool_calls for what each field means.
PENDING_APPROVALS: dict[str, dict[str, Any]] = {}


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


def drive_agent_loop(
    loop: Generator[dict[str, Any], None, dict[str, Any] | None],
    session_id: str,
    mode: str,
    workdir: Path,
) -> Generator[str, None, None]:
    """Formats a run_agent_loop generator's events as SSE lines, and records
    (or clears) PENDING_APPROVALS based on what it returns when it finishes.

    A plain `for event in loop:` can't see that return value - Python
    discards it - so this drives the generator by hand with next()/
    StopIteration instead, the same thing a for-loop does under the hood.
    """

    while True:
        try:
            event = next(loop)
        except StopIteration as stop:
            pending = stop.value
            if pending is not None:
                PENDING_APPROVALS[session_id] = {**pending, "mode": mode, "workdir": str(workdir)}
            else:
                PENDING_APPROVALS.pop(session_id, None)
            return
        yield f"data: {json.dumps(event)}\n\n"


def sse_stream(session_id: str, mode: str, workdir: Path) -> Generator[str, None, None]:
    history = SESSIONS[session_id]

    try:
        yield from drive_agent_loop(run_agent_loop(history, mode, workdir), session_id, mode, workdir)
    except Exception as error:  # noqa: BLE001 - last line of defense so the
        # SSE stream always terminates with an event the client can render,
        # instead of dying mid-response and leaving the frontend hanging.
        print(f"Unhandled error in agent loop: {error}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(error)})}\n\n"


def respond_stream(session_id: str, decision: Literal["approve", "deny"]) -> Generator[str, None, None]:
    pending = PENDING_APPROVALS.get(session_id)
    if pending is None:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No pending approval for this session'})}\n\n"
        return

    history = SESSIONS[session_id]
    mode = pending["mode"]
    workdir = Path(pending["workdir"])
    tc, *rest = pending["tool_calls"]
    tool_input = json.loads(tc["arguments"]) if tc["arguments"] else {}

    try:
        if decision == "approve":
            output = execute_tool(tc["name"], tool_input, mode, workdir)
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(output)})
            yield f"data: {json.dumps({'type': 'tool_result', 'tool_call_id': tc['id'], 'output': output})}\n\n"
        else:
            history.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": "User denied this action; it was not run."}
            )
            yield f"data: {json.dumps({'type': 'tool_denied', 'tool_call_id': tc['id']})}\n\n"
    except ToolError as error:
        history.append({"role": "tool", "tool_call_id": tc["id"], "content": f"error: {error}"})
        yield f"data: {json.dumps({'type': 'tool_error', 'tool_call_id': tc['id'], 'error': str(error)})}\n\n"

    resume_state = {"tool_calls": rest, "step": pending["step"], "prior_len": pending["prior_len"]}

    try:
        loop = run_agent_loop(history, mode, workdir, resume=resume_state)
        yield from drive_agent_loop(loop, session_id, mode, workdir)
    except Exception as error:  # noqa: BLE001
        print(f"Unhandled error resuming agent loop: {error}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(error)})}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id in PENDING_APPROVALS:
        raise HTTPException(409, "Resolve the pending tool approval (/chat/respond) before sending a new message")

    workdir = resolve_workdir(req.cwd)

    history = SESSIONS.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})

    return StreamingResponse(sse_stream(req.session_id, req.mode, workdir), media_type="text/event-stream")


@app.post("/chat/respond")
def respond(req: RespondRequest):
    if req.session_id not in PENDING_APPROVALS:
        raise HTTPException(404, "No pending approval for this session")

    return StreamingResponse(respond_stream(req.session_id, req.decision), media_type="text/event-stream")


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return {"messages": SESSIONS.get(session_id, [])}
