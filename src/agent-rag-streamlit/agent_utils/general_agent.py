"""
general_agent.py
Agent for handling general conversation queries
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import GENERAL_PROMPT

logger = logging.getLogger(__name__)


class GeneralAgent:
    """Agent for handling general queries that don't require document retrieval"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("General agent initialized")
    
    def general_response(self, state: ChatState) -> Dict[str, Any]:
        """Handle queries where documents weren't relevant enough"""
        query = state["query"]
        max_score = state.get("max_relevance_score", 0.0)
        
        logger.info(f"Generating general response for: '{query[:50]}...'")
        logger.info(f"Document relevance was insufficient (score: {max_score:.3f})")
        
        try:
            # Provide context about why documents weren't used
            context = f"Retrieved documents had a maximum relevance score of {max_score:.3f}, which was below the threshold for reliable document-based answers."
            
            messages = GENERAL_PROMPT.format_messages(query=query, context=context)
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info("General response generated successfully")
            
            return {
                **state,
                "answer": answer,
                "retrieved_docs": []
            }
            
        except Exception as e:
            logger.error(f"General response error: {e}")
            return {
                **state,
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "retrieved_docs": []
            }