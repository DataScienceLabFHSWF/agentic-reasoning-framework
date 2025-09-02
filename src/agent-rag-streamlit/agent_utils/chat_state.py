"""
chat_state.py
State definition for the RAG chat workflow with ReAct support
"""

from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langchain.schema import Document


class ChatState(TypedDict):
    """State of the chat workflow with ReAct loop tracking"""
    messages: List[BaseMessage]
    query: str
    is_corpus_relevant: Optional[bool]  # Intent classification result
    intent_reasoning: Optional[str]     # Classification reasoning
    is_relevant: Optional[bool]         # Document relevance decision
    retrieved_docs: List[Any]           # All retrieved documents (initial + additional)
    max_relevance_score: Optional[float]
    reasoning_answer: Optional[str]     # Answer from reasoning agent
    summarized_answer: Optional[str]    # Summarized response from documents
    final_answer: Optional[str]         # Final succinct answer
    chat_history: Optional[List[Dict[str, str]]]
    followup_questions: Optional[List[str]]      # All follow-up questions generated during ReAct loop
    additional_retrieved_context: Optional[int]  # Number of additional documents retrieved
    react_iterations: Optional[int]              # Number of ReAct iterations performed