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
    is_relevant: Optional[bool]
    retrieved_docs: List[Any]
    max_relevance_score: Optional[float]
    reasoning_answer: Optional[str]     # Answer from reasoning agent
    summarized_answer: Optional[str]    # Summarized response from documents
    final_answer: Optional[str]         # Final succinct answer
    chat_history: Optional[List[Dict[str, str]]]