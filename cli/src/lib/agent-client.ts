// Talks to the FastAPI /chat SSE endpoint. Hand-rolled: fetch + a
// ReadableStream reader, no EventSource (it can't send POST bodies) and no
// SDK wrapping the protocol - identical to the browser client in
// frontend/src/lib/agent-client.ts. Node's native fetch (18+) implements the
// same WHATWG ReadableStream interface a browser does, so this code needed
// zero changes to run outside a browser.

export type AgentEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_call"; tool_call_id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_call_id: string; output: unknown }
  | { type: "tool_error"; tool_call_id: string; error: string }
  | { type: "tool_skipped"; tool_call_id: string; name: string }
  | { type: "done"; stop_reason: string }
  | { type: "error"; message: string }
  | { type: "retry"; attempt: number; reason: string };

const API_URL = "http://localhost:8000";

export async function streamChat(
  params: { sessionId: string; message: string; mode: "PLAN" | "BUILD" },
  onEvent: (event: AgentEvent) => void,
) {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: params.sessionId,
      message: params.message,
      mode: params.mode,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
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
