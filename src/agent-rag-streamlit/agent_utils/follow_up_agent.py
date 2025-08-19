"""
follow_up_agent.py
Agent for handling follow-up questions using conversation context
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import FOLLOW_UP_PROMPT

logger = logging.getLogger(__name__)


class FollowUpAgent:
    """Agent for handling follow-up questions based on conversation context"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("Follow-up agent initialized")
    
    def handle_follow_up(self, state: ChatState) -> Dict[str, Any]:
        """Handle follow-up questions using previous conversation context"""
        query = state["query"]
        previous_context = state.get("previous_context", "")
        chat_history = state.get("chat_history", [])
        
        logger.info(f"Handling follow-up question: '{query[:50]}...'")
        
        # If no previous context, fall back to general response
        if not previous_context and not chat_history:
            logger.warning("No previous context available for follow-up")
            return {
                **state,
                "answer": "Es tut mir leid, aber ich habe keinen vorherigen Kontext, auf den ich mich beziehen könnte. Könnten Sie Ihre Frage bitte spezifischer stellen?"
            }
        
        # Use chat history if previous_context is not available
        if not previous_context and chat_history:
            last_exchanges = chat_history[-2:]  # Last 2 exchanges for more context
            previous_context = "\n\n".join([
                f"Frage: {item['user']}\nAntwort: {item['assistant']}" 
                for item in last_exchanges
            ])
        
        try:
            messages = FOLLOW_UP_PROMPT.format_messages(
                query=query,
                previous_context=previous_context
            )
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info("Follow-up response generated successfully")
            
            return {
                **state,
                "answer": answer,
                "retrieved_docs": []  # No new retrieval for follow-ups
            }
            
        except Exception as e:
            logger.error(f"Follow-up response error: {e}")
            return {
                **state,
                "answer": f"Es trat ein Fehler bei der Verarbeitung Ihrer Nachfrage auf: {str(e)}",
                "retrieved_docs": []
            }
