// Talks to the FastAPI SSE endpoints. Hand-rolled: fetch + a
// ReadableStream reader, no EventSource (it can't send POST bodies) and no
// SDK wrapping the protocol - nearly identical to the browser client in
// frontend/src/lib/agent-client.ts (Node's native fetch implements the same
// WHATWG ReadableStream interface a browser does). Real differences: this
// sends `cwd` (a browser has no real filesystem to sandbox tools to), and
// it has a second endpoint for resuming a turn paused on a human decision -
// either an approve/deny, or an answer to a clarifying question the model
// asked (agent/graph.py's `ask_question`).

export type AgentEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_call"; tool_call_id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_call_id: string; output: unknown }
  | { type: "tool_error"; tool_call_id: string; error: string }
  | { type: "tool_skipped"; tool_call_id: string; name: string }
  | { type: "approval_required"; tool_call_id: string; name: string; input: unknown }
  | { type: "tool_denied"; tool_call_id: string }
  | { type: "clarification_required"; tool_call_id: string; question: string; options: string[] }
  | { type: "done"; stop_reason: string }
  | { type: "error"; message: string }
  | { type: "retry"; attempt: number; reason: string };

const API_URL = "http://localhost:8000";

async function readEventStream(response: Response, onEvent: (event: AgentEvent) => void) {
  if (!response.ok || !response.body) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const raw of events) {
      const line = raw.trim();
      if (!line.startsWith("data:")) continue;
      onEvent(JSON.parse(line.slice(5).trim()) as AgentEvent);
    }
  }
}

export async function streamChat(
  params: { sessionId: string; message: string; mode: "PLAN" | "BUILD"; cwd: string },
  onEvent: (event: AgentEvent) => void,
) {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: params.sessionId,
      message: params.message,
      mode: params.mode,
      cwd: params.cwd,
    }),
  });
  await readEventStream(response, onEvent);
}

// `value` is "approve"/"deny" for a tool-approval pause, or the exact
// option text picked for a clarification-question pause - the caller
// already knows which kind it's resolving from the event it received.
export async function resumeTurn(
  params: { sessionId: string; value: string },
  onEvent: (event: AgentEvent) => void,
) {
  const response = await fetch(`${API_URL}/chat/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: params.sessionId, value: params.value }),
  });
  await readEventStream(response, onEvent);
}
