"""
intent_agent.py
Agent for classifying user intent and determining query relevance to document corpus
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import INTENT_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


class IntentClassificationAgent:
    """Agent for classifying user intent and routing based on document relevance"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("Intent classification agent initialized")
    
    def classify_intent(self, state: ChatState) -> Dict[str, Any]:
        """Classify user intent and determine if query is relevant to document corpus"""
        query = state["query"]
        
        logger.info(f"Classifying intent for query: '{query[:50]}...'")
        
        try:
            messages = INTENT_CLASSIFICATION_PROMPT.format_messages(query=query)
            response = self.llm.invoke(messages)
            
            # Parse response to determine relevance
            response_text = response.content.lower().strip()
            is_corpus_relevant = "relevant" in response_text and "not relevant" not in response_text
            
            logger.info(f"Intent classification result: {'corpus-relevant' if is_corpus_relevant else 'general'}")
            logger.info(f"Classification reasoning: {response.content[:100]}...")
            
            return {
                **state,
                "is_corpus_relevant": is_corpus_relevant,
                "intent_reasoning": response.content
            }
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            # Default to corpus-relevant on error to avoid blocking legitimate queries
            return {
                **state,
                "is_corpus_relevant": True,
                "intent_reasoning": f"Error in classification: {str(e)}"
            }
    
    def route_decision(self, state: ChatState) -> str:
        """Decision function for conditional routing"""
        decision = "corpus_relevant" if state.get("is_corpus_relevant", True) else "general"
        logger.info(f"Intent routing decision: {decision}")
        return decision
