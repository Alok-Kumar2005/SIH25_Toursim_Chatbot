from langgraph.graph import MessagesState
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AIComponentState(MessagesState):
    """State class for AI Component, extending MessagesState to manage chat messages."""
    
    messages: Annotated[list[BaseMessage], add_messages]
    language: str = "English"
    unique_id: str

    total_tokens: int