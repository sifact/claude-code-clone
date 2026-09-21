"""Model client config. Swap base_url/api_key/MODEL here to change providers."""

import os

from openai import OpenAI

MAX_TOKENS = 4096
MAX_STEPS = 10  # hard stop so a confused model can't loop forever
MAX_VALIDATION_RETRIES = 2  # retries after the API rejects the model's own generation

# 150s: this model can genuinely take 100+ seconds to respond on a normal
# request, so a tighter timeout would misfire more often than it'd catch a
# real hang.
MODEL = os.environ.get("AGENT_MODEL", "grok-build-0.1")
client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"), timeout=150.0)
