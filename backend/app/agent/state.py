"""The graph's state: what's persisted per session (per `thread_id`) by the
checkpointer. `mode`/`workdir` ride alongside `messages` since tool
execution needs them and LangGraph state is just a plain dict we define.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: str
    workdir: str
