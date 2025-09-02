"""
summarizer_agent.py
Agent for summarizing reasoning responses to create user-friendly answers
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import SUMMARIZER_PROMPT

logger = logging.getLogger(__name__)


class SummarizerAgent:
    """Agent for generating user-friendly responses based on reasoning answers"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("Summarizer agent initialized")
    
    def summarize_response(self, state: ChatState) -> Dict[str, Any]:
        """Summarize the reasoning answer to create a user-friendly response"""
        query = state["query"]
        reasoning_answer = state.get("reasoning_answer", "")
        
        print("\n📝 SUMMARIZER AGENT")
        print("-" * 50)
        print(f"📏 Input Length: {len(reasoning_answer)} characters")
        
        logger.info(f"Summarizing reasoning answer for query: '{query[:50]}...'")
        
        if not reasoning_answer:
            logger.warning("No reasoning answer available for summarization")
            answer = "Keine Antwort vom Reasoning-System erhalten."
            print("❌ No reasoning answer to summarize")
            print("→→→ Proceeding with empty response...")
        else:
            try:
                logger.info(f"Reasoning answer length: {len(reasoning_answer)} characters")
                
                print("🔄 Processing summarization request...")
                messages = SUMMARIZER_PROMPT.format_messages(
                    query=query, 
                    reasoning_answer=reasoning_answer
                )
                response = self.llm.invoke(messages)
                answer = response.content
                
                print(f"✅ SUMMARIZATION COMPLETE")
                print(f"📏 Summary Length: {len(answer)} characters")
                print(f"🎯 Summary Preview: {answer[:200]}...")
                print("→→→ Proceeding to FINAL ANSWER AGENT...")
                
                logger.info("Summarization completed successfully")
                
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                answer = f"Fehler beim Zusammenfassen der Antwort: {str(e)}"
                print(f"❌ Summarization Error: {str(e)}")
                print("→→→ Proceeding with error message...")
        
        return {
            **state,
            "summarized_answer": answer
        }