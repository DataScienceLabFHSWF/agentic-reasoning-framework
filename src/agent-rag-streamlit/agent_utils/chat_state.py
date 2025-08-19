"""
chat_state.py
State definition for the RAG chat workflow
"""

from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langchain.schema import Document


class ChatState(TypedDict):
    """State of the chat workflow"""
    messages: List[BaseMessage]
    query: str
    is_corpus_relevant: Optional[bool]  # New field for intent classification
    intent_reasoning: Optional[str]     # New field for classification reasoning
    is_follow_up: Optional[bool]        # New field for follow-up detection
    follow_up_context: Optional[str]    # New field for follow-up context
    human_intervention: Optional[str]   # New field for human intervention commands
    is_relevant: Optional[bool]
    retrieved_docs: List[Any]
    max_relevance_score: Optional[float]
    answer: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    previous_context: Optional[str]     # New field for storing previous conversation context