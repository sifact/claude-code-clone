"""Model client config. Swap base_url/api_key/model here to change providers.

`ChatOpenAI` accepts the same base_url/api_key/model kwargs as the raw
`openai` client we used in the hand-rolled version (verified live - they're
aliases for `openai_api_base`/`openai_api_key`/`model_name`), and its errors
multiply-inherit from `openai`'s own exception hierarchy, so the same
`except APIError` / `isinstance(error, APIConnectionError)` handling in
graph.py works completely unchanged.
"""

import os

from langchain_openai import ChatOpenAI

MAX_TOKENS = 4096
MAX_VALIDATION_RETRIES = 2  # retries after the API rejects the model's own generation

# LangGraph's recursion_limit counts graph steps, not agent turns - each
# call_model -> run_one_tool round trip is at least 2 steps, so this is
# generous enough for roughly the same 10-turn budget the hand-rolled
# version used (MAX_STEPS there), padded for multi-tool-call turns.
RECURSION_LIMIT = 40

MODEL = os.environ.get("AGENT_MODEL", "grok-build-0.1")
llm = ChatOpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.environ.get("XAI_API_KEY"),
    model=MODEL,
    max_tokens=MAX_TOKENS,
    timeout=150.0,
    streaming=True,
)
