# nightcode-fastapi

A from-scratch reimplementation of [nightcode](../nightcode)'s AI agent loop, using
FastAPI + React instead of Hono + AI SDK, with **no agent framework** — the
tool-calling loop is hand-written so the mechanics are visible instead of
hidden behind `streamText()` / `.compile()`.

Billing, auth, and persistence are intentionally left out (sessions are an
in-memory dict). The only thing this project is about is: how does an LLM
agent loop actually work, end to end.

Runs against **xAI's Grok** (OpenAI-compatible API) as the sole provider.
Model: `grok-4.6`.

This went through several earlier versions worth knowing about, since the
project's history is part of the lesson: it started on Groq alone, then grew
a Gemini fallback for when Groq's rate limit got hit, then swapped Gemini for
OpenRouter after Gemini turned out to violate the "OpenAI-compatible" label
in several ways (`index: null` instead of Groq's integer index, wrong
`finish_reason`, a mandatory opaque `thought_signature`), then dropped the
fallback entirely to run on OpenRouter's free tier alone (which came with its
own real-world friction — deprecated model names, a rerank model mistakenly
pointed at the chat endpoint, and a slow free-tier model that could burn its
whole step budget without converging), and finally moved to xAI, whose wire
format turned out to match the OpenAI/Groq standard with no quirks. See the
debugging log below for how each of those was actually found — Groq and
Gemini are gone from the code now, but the lessons from hitting each
provider's rough edges live on there.

## Debugging log (real bugs hit while building this)

Worth reading if you want to see what agent-loop robustness actually looks
like in practice, not just in theory:

1. **Model hallucinated an out-of-schema tool name** (`cat`, then later
   `repo_browser.print_tree` — both real tool-naming conventions from other
   agentic training environments the model picked up a prior for). Groq's API
   rejected the request with `openai.APIError`, which was originally
   unhandled and silently killed the whole SSE stream. Fixed by catching
   `APIError` and yielding a clean `error` event instead of crashing.
2. **A plain missing file crashed the same way** — `read_file` on a
   nonexistent path let a raw `FileNotFoundError` escape instead of becoming
   a `ToolError`. Same failure shape as #1, different layer (filesystem, not
   the API). Fixed in `tools.py` by catching `OSError`/`UnicodeDecodeError`
   around tool dispatch.
3. **Groq's free-tier rate limit** (8,000 tokens/minute on `gpt-oss-120b`,
   back when Groq was the provider) got hit fast because the whole message
   history — including full tool output content — is resent on every turn
   with no trimming. That's a project-wide issue independent of provider:
   whichever API is behind `client` in `agent.py` today, this codebase still
   doesn't bound or trim history, so a long enough session can still hit a
   free-tier rate limit.
4. **A Gemini fallback (since removed) failed its own way** on the very
   first live test — the `finish_reason` and `thought_signature` issues
   described above. Both were only found by forcing the fallback path
   deterministically (breaking the primary provider on purpose) and reading
   the actual error Gemini returned, not by reasoning about it in the
   abstract. The lesson (verify a second "OpenAI-compatible" provider live,
   don't assume it matches the first) outlived the fallback mechanism itself.
5. **Rejected tool calls left the model stuck repeating the same mistake.**
   When the API rejects the model's own generation (e.g. it calls `bash`
   without the required `command` argument), there's no `tool_call_id` to
   attach a normal tool-error response to - the whole completion got
   rejected before one existed. Originally this just ended the turn with an
   `error` event, so the model never learned anything failed, and the next
   similar user message would often trigger the exact same mistake: a dead
   end the user had to work around by hand. Fixed by giving the loop a
   bounded number of automatic retries (`MAX_VALIDATION_RETRIES`) that inject
   a corrective message ("your last attempt was rejected: `<reason>`, try
   again") into history before re-asking - turning it into a self-correcting
   loop instead of a hard stop. Verified with a scripted fake client that
   fails exactly twice then succeeds, confirming both the recovery path and
   the exhaustion-terminates-cleanly path. This survived the later removal of
   the multi-provider fallback unchanged, since it was always about the
   model's own bad generations, not about which provider served them.
6. **No client timeout, and the model kept re-reading the same files without
   converging on a fix.** Reproduced live by asking the agent to fix a
   failing test: it read `App.js` and `App.test.js`, then read both again,
   then again - 6 of its first 11 tool calls were pure repeats, directly
   against its own system prompt's "never re-read a file you've already
   read" rule, before eventually giving up with a 105-second gap of total
   silence (no timeout was configured, so the client just waited) and no
   fix. Two separate problems, two separate fixes: (a) `OpenAI(..., timeout=150.0)`
   now bounds every request - chosen deliberately high after confirming this
   free-tier model can legitimately take 100+ seconds to respond on a
   request that isn't actually hung, so a tighter timeout would misfire on
   normal responses more often than it'd catch a real hang; (b)
   `already_called()` in `agent/dedup.py` scans history for an identical prior
   read-only tool call (invalidated by any `write_file`/`edit_file`/`bash`
   call since, since the filesystem may have changed) and short-circuits
   repeats as a `tool_skipped` event instead of re-executing them - the
   model can't be relied on to police its own instruction, so it's enforced
   in code. Re-verified live: the same repeated `read_file` calls that
   previously executed twice more now get caught and skipped, and the model
   moves on to other actions (like actually running the test suite) instead
   of looping.

## Architecture

```
backend/    FastAPI. POST /chat streams Server-Sent Events.
  app/
    main.py             FastAPI app, SSE endpoint, in-memory sessions
    system_prompt.py    PLAN/BUILD mode-conditional prompt
    agent/              the loop itself — read loop.py first
      loop.py             run_agent_loop: ask model, run tools, repeat
      provider.py         model client config (base_url, model, timeout)
      dedup.py            skips a repeated read-only tool call
    tools/              tool schemas + sandboxed execution (server-side)
      schemas.py          the 7 tool definitions the model sees
      dispatch.py         routes a tool call by name to its handler
      filesystem.py       read_file/list_directory/glob/grep/write_file/edit_file
      shell.py            bash
      sandbox.py          shared path/size-limit helpers, ToolError

frontend/   React + Vite, styled as a terminal (dark, monospace, prompt-style
            input) even though it's rendered in a browser, not a real TTY.
  src/
    lib/agent-client.ts  fetch + ReadableStream SSE parsing (no EventSource,
                          it can't send POST bodies)
    App.tsx              terminal UI + event handling
```

Each backend file is small enough to read start to finish in one sitting.
`agent/loop.py` is the one file worth reading closely — everything else
supports it.

Tool execution lives on the **server**, not the client — unlike nightcode's
CLI, which executes tools locally because it has real filesystem access to
your terminal session. A browser can't touch a filesystem, so tools here run
sandboxed to `backend/workspace/`.

## Running it

```bash
# backend
cd backend
cp .env.example .env   # fill in XAI_API_KEY (console.x.ai)
uv run uvicorn app.main:app --reload --port 8000

# frontend (separate terminal)
cd frontend
npm run dev
```

Open the Vite dev URL (usually `http://localhost:5173`). Ask the agent to
explore or edit files in `backend/workspace/` and watch the raw tool-calling
loop play out.

## What to compare against nightcode

| Concept | nightcode | here |
|---|---|---|
| Agent loop | `streamText({ tools })` + `sendAutomaticallyWhen` (AI SDK) | explicit `for` loop in `agent/loop.py`, manually appending tool result messages |
| Tool execution | client-side (CLI has fs access) | server-side, sandboxed to `workspace/` |
| Mode gating | `getToolContracts(mode)` (shared) + re-checked in `local-tools.ts` | `get_tool_schemas(mode)` + re-checked in `execute_tool` |
| Model provider | Anthropic/OpenAI via AI SDK's unified interface | xAI/Grok (OpenAI-compatible) via the raw `openai` client |
| Tool-call wire format | discrete `tool_use` content blocks (Anthropic-style) | incremental JSON fragments keyed by index, reassembled in `agent/loop.py` |
| Streaming protocol | AI SDK's UIMessage stream | hand-rolled SSE, one JSON event per line |
| Step limit | implicit (`lastAssistantMessageIsCompleteWithToolCalls` loop) | explicit `MAX_STEPS` in `agent/provider.py` |

## Next steps (per the learning plan)

1. **This repo** — raw loop, no framework. ← you are here
2. Rebuild the same thing in LangGraph, with LangSmith tracing.
3. A pass through Semantic Kernel/AutoGen (Microsoft's own agent stack).
