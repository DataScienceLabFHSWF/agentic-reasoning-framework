"""
router_agent.py
Router agent for determining if retrieved documents are relevant enough for RAG
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)


class RouterAgent:
    """Agent for routing queries based on document relevance scores"""
    
    def __init__(self, llm: BaseChatModel, relevance_threshold: float = 0.5):
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        logger.info(f"Router agent initialized with threshold: {relevance_threshold}")
    
    def route_query(self, state: ChatState) -> Dict[str, Any]:
        """Route query based on document relevance scores"""
        query = state["query"]
        max_score = state.get("max_relevance_score", 0.0)
        
        print("\n🔀 ROUTER AGENT")
        print("-" * 50)
        print(f"📊 Max Relevance Score: {max_score:.3f}")
        print(f"🎯 Threshold: {self.relevance_threshold}")
        
        logger.info(f"Router evaluating relevance: score={max_score:.3f}, threshold={self.relevance_threshold}")
        
        # Primary decision based on relevance score
        is_relevant = max_score >= self.relevance_threshold
        
        if is_relevant:
            print("✅ DECISION: Documents meet relevance threshold")
            print("→→→ Routing to REASONING AGENT...")
            logger.info("Documents meet relevance threshold - using RAG")
        else:
            print("❌ DECISION: Documents below relevance threshold")
            print("→→→ Routing to GENERAL RESPONSE...")
            logger.info("Documents below relevance threshold - using general response")
            
            # Use LLM to provide context about why docs weren't relevant
            try:
                messages = ROUTER_PROMPT.format_messages(
                    query=query, 
                    max_score=max_score, 
                    threshold=self.relevance_threshold
                )
                response = self.llm.invoke(messages)
                logger.info(f"Router reasoning: {response.content[:100]}...")
            except Exception as e:
                logger.error(f"Router LLM error: {e}")
        
        return {
            **state,
            "is_relevant": is_relevant
        }
    
    def route_decision(self, state: ChatState) -> str:
        """Decision function for conditional routing"""
        decision = "relevant" if state["is_relevant"] else "not_relevant"
        logger.info(f"Final routing decision: {decision}")
        return decision