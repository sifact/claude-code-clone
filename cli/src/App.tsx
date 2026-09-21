import { randomUUID } from "node:crypto";
import { useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import SelectInput from "ink-select-input";
import { resumeTurn, streamChat, type AgentEvent } from "./lib/agent-client.js";

type Mode = "PLAN" | "BUILD";

type PendingApproval = { toolCallId: string; name: string; input: unknown };
type PendingClarification = { toolCallId: string; question: string; options: string[] };

type LogEntry =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string; streaming: boolean }
  | { kind: "tool_call"; id: string; name: string; input: unknown }
  | { kind: "tool_result"; id: string; output: unknown }
  | { kind: "tool_error"; id: string; error: string }
  | { kind: "tool_skipped"; id: string; name: string }
  | { kind: "tool_denied"; id: string }
  | { kind: "clarification"; id: string; question: string }
  | { kind: "clarification_answered"; id: string; answer: string }
  | { kind: "system"; id: string; text: string };

const SESSION_ID = randomUUID();

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
    { kind: "system", id: "boot-cwd", text: `operating in ${process.cwd()} -- tools run for real here` },
  ]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("BUILD");
  const [busy, setBusy] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [pendingClarification, setPendingClarification] = useState<PendingClarification | null>(null);
  const streamingIdRef = useRef<string | null>(null);

  function appendLog(entry: LogEntry) {
    setLog((prev) => [...prev, entry]);
  }

  function handleEvent(event: AgentEvent) {
    if (event.type === "text_delta") {
      if (!streamingIdRef.current) {
        const id = randomUUID();
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

    if (event.type === "approval_required") {
      streamingIdRef.current = null;
      appendLog({ kind: "tool_call", id: event.tool_call_id, name: event.name, input: event.input });
      setPendingApproval({ toolCallId: event.tool_call_id, name: event.name, input: event.input });
      return;
    }

    if (event.type === "tool_denied") {
      appendLog({ kind: "tool_denied", id: `${event.tool_call_id}-denied` });
      return;
    }

    if (event.type === "clarification_required") {
      streamingIdRef.current = null;
      appendLog({ kind: "clarification", id: event.tool_call_id, question: event.question });
      setPendingClarification({ toolCallId: event.tool_call_id, question: event.question, options: event.options });
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
      appendLog({ kind: "system", id: randomUUID(), text: `error: ${event.message}` });
      return;
    }

    if (event.type === "retry") {
      appendLog({
        kind: "system",
        id: randomUUID(),
        text: `⚠ tool call rejected, retrying (attempt ${event.attempt})…`,
      });
    }
  }

  async function handleSubmit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy || pendingApproval || pendingClarification) return;

    appendLog({ kind: "user", id: randomUUID(), text: trimmed });
    setInput("");
    setBusy(true);
    streamingIdRef.current = null;

    try {
      await streamChat({ sessionId: SESSION_ID, message: trimmed, mode, cwd: process.cwd() }, handleEvent);
    } catch (error) {
      appendLog({
        kind: "system",
        id: randomUUID(),
        text: `error: ${error instanceof Error ? error.message : String(error)}`,
      });
    } finally {
      setBusy(false);
    }
  }

  async function resume(value: string) {
    setPendingApproval(null);
    setPendingClarification(null);
    setBusy(true);
    streamingIdRef.current = null;

    try {
      await resumeTurn({ sessionId: SESSION_ID, value }, handleEvent);
    } catch (error) {
      appendLog({
        kind: "system",
        id: randomUUID(),
        text: `error: ${error instanceof Error ? error.message : String(error)}`,
      });
    } finally {
      setBusy(false);
    }
  }

  function resolveApproval(decision: "approve" | "deny") {
    if (!pendingApproval) return;
    void resume(decision);
  }

  function resolveClarification(answer: string) {
    if (!pendingClarification) return;
    appendLog({ kind: "clarification_answered", id: `${pendingClarification.toolCallId}-answer`, answer });
    void resume(answer);
  }

  // No mouse in a terminal: the mode toggle (a clickable button in the
  // browser) is a keybinding, and so is the approve/deny prompt. The
  // clarification prompt's arrow-key navigation is handled by SelectInput
  // itself below, not here - this only needs to stay out of its way.
  useInput((rawInput, key) => {
    if (pendingClarification) return;
    if (pendingApproval) {
      const answer = rawInput.toLowerCase();
      if (answer === "y" || key.return) resolveApproval("approve");
      else if (answer === "n" || key.escape) resolveApproval("deny");
      return;
    }
    if (key.tab && !busy) {
      setMode((m) => (m === "BUILD" ? "PLAN" : "BUILD"));
    }
  });

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="gray" paddingX={1}>
      <Box justifyContent="space-between" marginBottom={1}>
        <Text bold>nightcode-fastapi</Text>
        <Text color={mode === "PLAN" ? "cyan" : "green"} bold>
          [{mode}]
        </Text>
      </Box>

      <Box flexDirection="column">
        {log.map((entry) => (
          <LogLine key={entry.id} entry={entry} />
        ))}
        {busy && !streamingIdRef.current && !pendingApproval && !pendingClarification && (
          <Text dimColor>…thinking</Text>
        )}
      </Box>

      {pendingClarification ? (
        <Box marginTop={1} flexDirection="column">
          <Text color="magenta" bold>
            ? {pendingClarification.question}
          </Text>
          <SelectInput
            items={pendingClarification.options.map((option) => ({ label: option, value: option }))}
            onSelect={(item) => resolveClarification(item.value)}
          />
        </Box>
      ) : pendingApproval ? (
        <Box marginTop={1} flexDirection="column">
          <Text color="yellow" bold>
            ⚠ approve {pendingApproval.name} {formatToolInput(pendingApproval.input)}?
          </Text>
          <Text dimColor>y / enter: approve · n / esc: deny</Text>
        </Box>
      ) : (
        <>
          <Box marginTop={1}>
            <Text color="green" bold>
              {mode === "PLAN" ? "plan" : "build"}{" "}❯{" "}
            </Text>
            <TextInput
              value={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              focus={!busy}
              placeholder={busy ? "streaming…" : "ask the agent to do something"}
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>tab: toggle mode · enter: send · ctrl+c: quit</Text>
          </Box>
        </>
      )}
    </Box>
  );
}

function LogLine({ entry }: { entry: LogEntry }) {
  switch (entry.kind) {
    case "user":
      return (
        <Text>
          <Text color="green">❯</Text> {entry.text}
        </Text>
      );
    case "assistant":
      return <Text>{entry.text}{entry.streaming ? "▌" : ""}</Text>;
    case "tool_call":
      return (
        <Text color="yellow">
          ⚙ {entry.name} <Text dimColor>{formatToolInput(entry.input)}</Text>
        </Text>
      );
    case "tool_result":
      return (
        <Text color="cyan">
          ← <Text dimColor>{formatToolOutput(entry.output)}</Text>
        </Text>
      );
    case "tool_error":
      return <Text color="red">✕ {entry.error}</Text>;
    case "tool_skipped":
      return <Text dimColor>↻ skipped repeated {entry.name} call (already have this result)</Text>;
    case "tool_denied":
      return <Text color="red">✕ denied — not executed</Text>;
    case "clarification":
      return <Text color="magenta">? {entry.question}</Text>;
    case "clarification_answered":
      return (
        <Text>
          <Text color="magenta">↳</Text> {entry.answer}
        </Text>
      );
    case "system":
      return (
        <Text dimColor italic>
          {entry.text}
        </Text>
      );
  }
}
