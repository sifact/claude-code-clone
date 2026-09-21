# nightcode-fastapi

A from-scratch reimplementation of [nightcode](../nightcode)'s AI agent loop, using
FastAPI + React/Ink instead of Hono + AI SDK.

**This branch (`feature/langgraph`) is step 2 of the learning plan**: the
same agent loop rebuilt on LangGraph instead of hand-written. `main` and the
earlier `feature/*` branches have the hand-rolled version (`agent/loop.py`,
a plain `for` loop with no framework); this branch replaces `agent/`
internals with a `StateGraph`. `tools/`, `system_prompt.py`, and the browser
client are untouched; the CLI gained one new feature this branch's mechanism
made easy to add (clarification questions, arrow-key answered - see below),
which is the only reason it isn't identical too. That's what makes the core
comparison a fair one rather than a rewrite: same loop behavior, verified
live here the same way it was on the hand-rolled branches, different
mechanism underneath. See "Hand-rolled vs LangGraph" below for what that
swap actually bought.

Billing, auth, and persistence are intentionally left out (sessions are an
in-memory dict - or, on this branch, an in-memory checkpointer, same idea).
The only thing this project is about is: how does an LLM agent loop actually
work, end to end.

Runs against **xAI's Grok** (OpenAI-compatible API) as the sole provider.
Model: `grok-build-0.1` — picked after comparing prices across xAI's `/models`
endpoint (it's the cheapest chat model in the catalog: half `grok-4.6`'s
prompt cost, a third its completion cost) and confirming live that it still
supports tool calling correctly.

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
7. **Porting `already_called()` to LangGraph's state introduced a fresh bug
   in the exact same spot** - the first `list_directory` call on a brand new
   thread got flagged as a duplicate *of itself*. Cause: by the time
   `run_one_tool` runs, `state["messages"]` already includes the AIMessage
   that requested the call being checked, so the dedup scan found a
   "prior" match that was really just itself. The hand-rolled version dodged
   this with a `prior_messages = messages[:-1]` snapshot taken *before* the
   assistant message was appended; that exact slice isn't safe here because
   a multi-tool-call turn can have several `run_one_tool` invocations after
   one `call_model` call, so `[-1]` isn't reliably "the triggering message."
   Fixed by identity instead of position: `_next_unanswered_tool_call` now
   returns the triggering `AIMessage` itself, and the dedup check filters it
   out by object identity (`m is not triggering_message`) rather than by
   slicing. Same bug class as #6, different framework, caught the same way -
   by testing live instead of trusting that the port was equivalent.

## Architecture

```
backend/    FastAPI. POST /chat streams Server-Sent Events.
  app/
    main.py             FastAPI app, SSE endpoints - no SESSIONS/PENDING_APPROVALS
                         dicts on this branch, the checkpointer owns that now
    system_prompt.py    PLAN/BUILD mode-conditional prompt (unchanged)
    agent/              the graph itself — read graph.py first
      graph.py            builds the StateGraph: call_model + run_one_tool
                           nodes, conditional routing, interrupt()-based
                           approval, InMemorySaver checkpointer
      state.py            the graph's state shape (messages, mode, workdir)
      provider.py          ChatOpenAI client config (same base_url/model as
                           the hand-rolled version's raw openai client)
      dedup.py            skips a repeated read-only tool call (same idea as
                           the hand-rolled version, adapted for BaseMessage)
      approval.py         which tools pause for a human decision (CLI only,
                           unchanged - framework-agnostic either way)
    tools/              tool schemas + sandboxed execution (server-side) -
                         same as the hand-rolled branches, except one
                         addition below
      schemas.py          the 8 tool definitions the model sees - 7 unchanged,
                           plus `ask_question` (added on this branch: see
                           "Clarification questions" below)
      dispatch.py         routes a tool call by name to its handler -
                           `ask_question` never reaches this; agent/graph.py
                           intercepts it before dispatch
      filesystem.py       read_file/list_directory/glob/grep/write_file/edit_file
      shell.py            bash
      sandbox.py          shared path/size-limit helpers, ToolError

frontend/   React + Vite, styled as a terminal (dark, monospace, prompt-style
            input) even though it's rendered in a browser, not a real TTY.
  src/
    lib/agent-client.ts  fetch + ReadableStream SSE parsing (no EventSource,
                          it can't send POST bodies)
    App.tsx              terminal UI + event handling

cli/        A real terminal client via Ink (React renderer for terminals,
            same idea as nightcode's own OpenTUI) instead of a browser.
  src/
    lib/agent-client.ts  same SSE client as the browser's, plus `cwd` and
                          `/chat/respond` for resuming a paused turn
    App.tsx              Ink UI: Box/Text instead of div/CSS, Tab to toggle
                          mode, ink-select-input for clarification answers
                          (arrow keys + enter) instead of a clickable button
```

Each backend file is small enough to read start to finish in one sitting.
`agent/graph.py` is the one file worth reading closely — everything else
supports it.

**Tool execution still happens on the server**, not the client — every tool
call goes through `tools/dispatch.py`, same as the hand-rolled version,
regardless of which frontend is asking. Sandboxing is still per-request: the
CLI sends its own `process.cwd()` (`state["workdir"]`), the browser omits it
and falls back to `DEFAULT_WORKDIR`. None of that changed on this branch -
it's entirely inside `tools/`, which this branch doesn't touch.

**Human-in-the-loop approval (CLI only) — now via `interrupt()`.** This is
the part LangGraph actually buys something real for. The hand-rolled version
needed a hand-built `PENDING_APPROVALS` dict, a second endpoint that
manually replayed remaining tool calls, and threading a `resume` state
through `run_agent_loop` to pick back up correctly. Here, `run_one_tool`
just calls `interrupt({...})` when a tool needs approval (`agent/graph.py`);
LangGraph's checkpointer persists *everything* - full state, which node was
running, all of it - the instant that happens, and `POST /chat/respond`
resumes with nothing more than `agent_graph.stream(Command(resume=decision),
config)`. No custom state dict, no manual replay logic.

The one thing that isn't free: `interrupt()` re-runs its node **from the
top** on resume, so any side effect *before* the interrupt call in that same
node would silently re-execute too. That's why `run_one_tool` handles
exactly one tool call per invocation - via a graph edge that loops back to
itself while there's more to do, not a Python `for` loop over all of them -
so every invocation calls `interrupt()` at most once, before any side
effect, with nothing earlier in that same call for a resume to replay.
Verified live end-to-end through the real `/chat` → `/chat/respond` HTTP
round-trip: approve executes and the graph correctly continues (including
pausing again on a second approval-required call reached in that
continuation, and the model persistently trying an *alternative* tool after
one denial, correctly caught by the same gate again); deny skips execution
and the model adapts, same as the hand-rolled version.

**Clarification questions (CLI only) — the same `interrupt()`, a different
reason to use it.** `ask_question` (`tools/schemas.py`) isn't a real tool in
the filesystem/shell sense - it's available in *both* PLAN and BUILD mode,
and `run_one_tool` special-cases it before any of the approval/dedup/execute
logic even runs. When the model calls it, `interrupt({"kind":
"clarification", "question": ..., "options": [...]})` pauses the graph the
same way an approval does, but the resume value isn't approve/deny - it's
the exact option text the user picked, which becomes that tool call's
result directly. `main.py` tells the two kinds of pause apart by the
`"kind"` field on the interrupt's value and emits a distinct SSE event
(`clarification_required` vs `approval_required`); `/chat/respond`'s
request body was generalized from `decision: "approve"|"deny"` to a plain
`value: str` to carry either one, since the frontend always knows which
kind it's answering from the event it received. In the CLI, this renders as
an `ink-select-input` list - arrow keys to move, enter to pick - replacing
the normal input row while it's showing, the same way the approval prompt
does. Verified live: the model asking a real clarifying question with
sensible options, the answer correctly becoming that tool's result and the
model referencing it in its next response, `ask_question` working
identically in PLAN mode (it's read-only in spirit even though it's not in
`tools/dispatch.py` at all), and the `409` guard correctly blocking a new
message while a clarification is still pending - same as it already did for
approvals.

## Running it

```bash
# backend
cd backend
cp .env.example .env   # fill in XAI_API_KEY (console.x.ai)
                        # LANGSMITH_* is optional on this branch - ambient
                        # env vars, no code changes needed to enable tracing
uv run uvicorn app.main:app --reload --port 8000

# browser client (separate terminal)
cd frontend
npm run dev

# or the terminal client instead (separate terminal, from wherever you want it to operate)
cd cli
npm run dev
```

Open the Vite dev URL (usually `http://localhost:5173`) for the browser
version. Ask the agent to explore or edit files and watch the raw
tool-calling loop play out — in `backend/workspace/` for the browser client,
or wherever you launched `cli/` from for the terminal client.

## Hand-rolled vs LangGraph

The direct comparison this branch exists for. "Hand-rolled" means `main` /
the earlier `feature/*` branches; "LangGraph" is this branch. Same
behavior, same tests passing, same frontends untouched either way.

| Concept | Hand-rolled | LangGraph |
|---|---|---|
| Loop structure | explicit `for` loop in `agent/loop.py`, manually appending messages | `StateGraph` with two nodes (`call_model`, `run_one_tool`) and conditional edges between them |
| Turn/session state | `SESSIONS` dict in `main.py`, mutated in place | `InMemorySaver` checkpointer, keyed by `thread_id` (= `session_id`) - `main.py` holds no state at all |
| Human-in-the-loop pause/resume | `PENDING_APPROVALS` dict + a hand-threaded `resume` param through `run_agent_loop`, `StopIteration.value` to hand back what's pending | `interrupt()` + `Command(resume=...)` - the checkpointer persists everything automatically |
| Tool-call wire format handling | manual: reassemble incremental JSON-string fragments keyed by index (`agent/loop.py`) | handled by `langchain-openai`'s `ChatOpenAI` - never touched directly |
| Retry on a rejected generation | explicit `attempt` loop + `isinstance(error, APIConnectionError)` branch | same explicit logic, same exception types (LangChain's wrapped errors multiply-inherit from `openai`'s own hierarchy - `except APIError` needed no changes) |
| Step budget | `MAX_STEPS`, a plain loop counter | `recursion_limit` on `graph.stream()` - counts graph steps, not turns, so it's not a 1:1 mapping |
| Token-level streaming | native - the raw `openai` client's `.stream()` | `llm.stream()` inside `call_model`, pushed out via `get_stream_writer()` on LangGraph's `"custom"` channel |
| Tracing | none built in | `LANGSMITH_TRACING=true` + an API key - zero code changes, purely ambient |

**What genuinely got simpler**: the entire human-in-the-loop mechanism.
`PENDING_APPROVALS`, the manual "apply the decision to the first pending
item, then continue the rest" logic, threading `resume` through the outer
step loop - all of that was custom-built to work around one constraint (an
SSE response can't wait for a keypress). `interrupt()` + a checkpointer
solves the identical constraint natively, and does it more capably (full
state, not just a hand-picked slice of it).

**What didn't get simpler, or even got a little more subtle**: multi-tool-
call turns. The hand-rolled version could safely loop over every tool call
in a Python `for` loop. Here, that same loop had to move to the *graph*
level (`run_one_tool` looping back to itself via an edge) specifically
because of how `interrupt()` replays a node on resume - a real constraint
the framework introduces that the hand-rolled version never had to think
about, because it never had this pause/resume superpower to begin with.

**What's most telling**: the exact same bug (a dedup check matching itself
because it ran too late relative to when the triggering message got
appended) showed up in *both* implementations, independently, at the same
point. Same root cause, different framework - the mechanics don't get
safer just because a framework is doing more of the surrounding work.

**What's genuinely new, not just easier**: clarification questions
(`ask_question`). This isn't in the hand-rolled version at all - it wasn't
just harder to build there, it wasn't attempted, because the whole
`PENDING_APPROVALS`/manual-resume apparatus was already bespoke enough for
one purpose (approve/deny). Once `interrupt()` existed as a general "pause
for any reason, resume with any value" primitive, adding a second, unrelated
use for it (a multiple-choice question instead of a yes/no) was a small,
contained change: one more tool definition, one more branch in
`run_one_tool`, one more case in the SSE translation. That's the real
argument for the framework - not that any single feature became impossible
to build by hand, but that the *second* feature built on the same mechanism
got cheap once the mechanism existed as a reusable primitive instead of a
one-off.

## Next steps (per the learning plan)

1. Raw loop, no framework — the hand-rolled branches (`main` and the earlier `feature/*`).
2. **This branch** — the same thing rebuilt on LangGraph, with LangSmith tracing. ← you are here
3. A pass through Semantic Kernel/AutoGen (Microsoft's own agent stack).
