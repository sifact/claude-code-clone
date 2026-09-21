import { useRef, useState } from "react";
import "./App.css";
import { streamChat, type AgentEvent } from "./lib/agent-client";

type Mode = "PLAN" | "BUILD";

type LogEntry =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string; streaming: boolean }
  | { kind: "tool_call"; id: string; name: string; input: unknown }
  | { kind: "tool_result"; id: string; output: unknown }
  | { kind: "tool_error"; id: string; error: string }
  | { kind: "tool_skipped"; id: string; name: string }
  | { kind: "system"; id: string; text: string };

const SESSION_ID = crypto.randomUUID();

function formatToolInput(input: unknown) {
  return JSON.stringify(input);
}

function formatToolOutput(output: unknown) {
  const text = typeof output === "string" ? output : JSON.stringify(output);
  return text.length > 400 ? `${text.slice(0, 400)}… (truncated)` : text;
}

export default function App() {
  const [log, setLog] = useState<LogEntry[]>([
    { kind: "system", id: "boot", text: "nightcode-fastapi -- hand-rolled agent loop, no framework" },
  ]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("BUILD");
  const [busy, setBusy] = useState(false);
  const streamingIdRef = useRef<string | null>(null);

  function appendLog(entry: LogEntry) {
    setLog((prev) => [...prev, entry]);
  }

  function handleEvent(event: AgentEvent) {
    if (event.type === "text_delta") {
      if (!streamingIdRef.current) {
        const id = crypto.randomUUID();
        streamingIdRef.current = id;
        appendLog({ kind: "assistant", id, text: event.text, streaming: true });
        return;
      }
      const id = streamingIdRef.current;
      setLog((prev) =>
        prev.map((entry) =>
          entry.kind === "assistant" && entry.id === id
            ? { ...entry, text: entry.text + event.text }
            : entry,
        ),
      );
      return;
    }

    if (event.type === "tool_call") {
      streamingIdRef.current = null;
      appendLog({ kind: "tool_call", id: event.tool_call_id, name: event.name, input: event.input });
      return;
    }

    if (event.type === "tool_result") {
      appendLog({ kind: "tool_result", id: `${event.tool_call_id}-out`, output: event.output });
      return;
    }

    if (event.type === "tool_error") {
      appendLog({ kind: "tool_error", id: `${event.tool_call_id}-err`, error: event.error });
      return;
    }

    if (event.type === "tool_skipped") {
      appendLog({ kind: "tool_skipped", id: `${event.tool_call_id}-skip`, name: event.name });
      return;
    }

    if (event.type === "done") {
      streamingIdRef.current = null;
      setLog((prev) =>
        prev.map((entry) => (entry.kind === "assistant" ? { ...entry, streaming: false } : entry)),
      );
      return;
    }

    if (event.type === "error") {
      streamingIdRef.current = null;
      setLog((prev) =>
        prev.map((entry) => (entry.kind === "assistant" ? { ...entry, streaming: false } : entry)),
      );
      appendLog({ kind: "system", id: crypto.randomUUID(), text: `error: ${event.message}` });
      return;
    }

    if (event.type === "retry") {
      appendLog({
        kind: "system",
        id: crypto.randomUUID(),
        text: `⚠ tool call rejected, retrying (attempt ${event.attempt})…`,
      });
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;

    appendLog({ kind: "user", id: crypto.randomUUID(), text });
    setInput("");
    setBusy(true);
    streamingIdRef.current = null;

    try {
      await streamChat({ sessionId: SESSION_ID, message: text, mode }, handleEvent);
    } catch (error) {
      appendLog({
        kind: "system",
        id: crypto.randomUUID(),
        text: `error: ${error instanceof Error ? error.message : String(error)}`,
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="terminal">
      <div className="terminal-titlebar">
        <span className="terminal-dots">
          <span className="dot dot-red" />
          <span className="dot dot-yellow" />
          <span className="dot dot-green" />
        </span>
        <span className="terminal-title">nightcode-fastapi</span>
        <button
          type="button"
          className={`mode-toggle mode-${mode.toLowerCase()}`}
          onClick={() => setMode((m) => (m === "BUILD" ? "PLAN" : "BUILD"))}
          disabled={busy}
        >
          [{mode}]
        </button>
      </div>

      <div className="terminal-body">
        {log.map((entry) => <LogLine key={entry.id} entry={entry} />)}
        {busy && !streamingIdRef.current && <div className="line line-system">…thinking</div>}
      </div>

      <form className="terminal-input-row" onSubmit={handleSubmit}>
        <span className="prompt">{mode === "PLAN" ? "plan" : "build"}&nbsp;❯</span>
        <input
          autoFocus
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
          placeholder={busy ? "streaming…" : "ask the agent to do something"}
          spellCheck={false}
        />
      </form>
    </div>
  );
}

function LogLine({ entry }: { entry: LogEntry }) {
  switch (entry.kind) {
    case "user":
      return (
        <div className="line line-user">
          <span className="prompt-inline">❯</span> {entry.text}
        </div>
      );
    case "assistant":
      return (
        <div className="line line-assistant">
          {entry.text}
          {entry.streaming && <span className="cursor" />}
        </div>
      );
    case "tool_call":
      return (
        <div className="line line-tool-call">
          ⚙ {entry.name} <span className="dim">{formatToolInput(entry.input)}</span>
        </div>
      );
    case "tool_result":
      return (
        <div className="line line-tool-result">
          ← <span className="dim">{formatToolOutput(entry.output)}</span>
        </div>
      );
    case "tool_error":
      return (
        <div className="line line-tool-error">
          ✕ {entry.error}
        </div>
      );
    case "tool_skipped":
      return (
        <div className="line line-system">
          ↻ skipped repeated {entry.name} call (already have this result)
        </div>
      );
    case "system":
      return <div className="line line-system">{entry.text}</div>;
  }
}
