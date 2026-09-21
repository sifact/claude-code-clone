"""FastAPI entrypoint. One endpoint: POST /chat, streamed as SSE.

Sessions are an in-memory dict (list of OpenAI-format chat messages per
session id) - no database. That's a deliberate simplification to keep this
phase focused on the agent loop itself, not persistence.
"""

import json
from collections.abc import Generator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from .agent import run_agent_loop  # noqa: E402

app = FastAPI(title="nightcode-fastapi")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "BUILD"


def sse_stream(session_id: str, mode: str) -> Generator[str, None, None]:
    history = SESSIONS[session_id]

    try:
        for event in run_agent_loop(history, mode):
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as error:  # noqa: BLE001 - last line of defense so the
        # SSE stream always terminates with an event the client can render,
        # instead of dying mid-response and leaving the frontend hanging.
        print(f"Unhandled error in agent loop: {error}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(error)})}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    history = SESSIONS.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})

    return StreamingResponse(sse_stream(req.session_id, req.mode), media_type="text/event-stream")


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return {"messages": SESSIONS.get(session_id, [])}
