"""
chat_state.py
State definition for the RAG chat workflow
"""

from typing import Dict, Any, List, TypedDict
from langchain_core.messages import BaseMessage
from langchain.schema import Document


class ChatState(TypedDict):
    """State of the chat workflow"""
    messages: List[BaseMessage]
    query: str
    is_relevant: bool
    retrieved_docs: List[Document]
    max_relevance_score: float
    answer: str
    chat_history: List[Dict[str, str]]