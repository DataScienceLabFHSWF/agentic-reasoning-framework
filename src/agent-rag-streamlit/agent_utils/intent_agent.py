"""
intent_agent.py
Agent for classifying user intent and determining query relevance to document corpus
"""

import logging
from typing import Dict, Any, Optional
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
        chat_history = state.get("chat_history", [])
        
        logger.info(f"Classifying intent for query: '{query[:50]}...'")
        
        # Check for human intervention commands
        human_intervention = self._detect_human_intervention(query)
        if human_intervention:
            logger.info(f"Human intervention detected: {human_intervention}")
            return {
                **state,
                "human_intervention": human_intervention,
                "is_corpus_relevant": True,  # Let workflow handle it
                "is_follow_up": False,
                "intent_reasoning": f"Human intervention command: {human_intervention}"
            }
        
        try:
            # Format chat history for context
            history_text = ""
            if chat_history:
                recent_history = chat_history[-3:]  # Last 3 exchanges
                history_text = "\n".join([
                    f"Benutzer: {item['user']}\nAssistent: {item['assistant']}" 
                    for item in recent_history
                ])
            
            messages = INTENT_CLASSIFICATION_PROMPT.format_messages(
                query=query, 
                chat_history=history_text
            )
            response = self.llm.invoke(messages)
            
            # Parse response to determine classification
            response_text = response.content.lower().strip()
            
            if "follow_up" in response_text:
                is_follow_up = True
                is_corpus_relevant = True
                classification = "follow-up"
            elif "relevant" in response_text and "not relevant" not in response_text:
                is_follow_up = False
                is_corpus_relevant = True
                classification = "corpus-relevant"
            else:
                is_follow_up = False
                is_corpus_relevant = False
                classification = "general"
            
            logger.info(f"Intent classification result: {classification}")
            logger.info(f"Classification reasoning: {response.content[:100]}...")
            
            # Store previous context for follow-ups
            previous_context = ""
            if chat_history:
                last_exchange = chat_history[-1]
                previous_context = f"Letzte Frage: {last_exchange['user']}\nLetzte Antwort: {last_exchange['assistant']}"
            
            return {
                **state,
                "is_corpus_relevant": is_corpus_relevant,
                "is_follow_up": is_follow_up,
                "previous_context": previous_context,
                "intent_reasoning": response.content,
                "human_intervention": None
            }
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            # Default to corpus-relevant on error to avoid blocking legitimate queries
            return {
                **state,
                "is_corpus_relevant": True,
                "is_follow_up": False,
                "intent_reasoning": f"Error in classification: {str(e)}",
                "human_intervention": None
            }
    
    def _detect_human_intervention(self, query: str) -> Optional[str]:
        """Detect human intervention commands in the query"""
        query_lower = query.lower().strip()
        
        # Commands for forcing RAG
        if any(cmd in query_lower for cmd in ["force rag", "force search", "search anyway", "!rag"]):
            return "force_rag"
        
        # Commands for using only context
        if any(cmd in query_lower for cmd in ["use context", "only context", "no search", "!context"]):
            return "use_context"
        
        # Commands for general response
        if any(cmd in query_lower for cmd in ["general", "no rag", "!general"]):
            return "general"
        
        return None
    
    def route_decision(self, state: ChatState) -> str:
        """Decision function for conditional routing"""
        # Check for human intervention first
        if state.get("human_intervention"):
            decision = "human_intervention"
        elif state.get("is_follow_up", False):
            decision = "follow_up"
        elif state.get("is_corpus_relevant", True):
            decision = "corpus_relevant"
        else:
            decision = "general"
        
        logger.info(f"Intent routing decision: {decision}")
        return decision
