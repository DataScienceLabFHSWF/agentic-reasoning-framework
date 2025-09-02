"""
final_answer_agent.py
Agent for generating a succinct, final answer from a summarized response.
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import FINAL_ANSWER_PROMPT

logger = logging.getLogger(__name__)


class FinalAnswerAgent:
    """
    Agent to produce a final, one-or-two-word/entity/number answer for
    evaluation purposes.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("Final answer agent initialized")
    
    def get_final_answer(self, state: ChatState) -> Dict[str, Any]:
        """
        Generates a succinct answer based on the summarized response.
        """
        query = state["query"]
        summarized_answer = state.get("summarized_answer", "")
        
        print("\n🎯 FINAL ANSWER AGENT")
        print("-" * 50)
        print(f"📏 Input Length: {len(summarized_answer)} characters")
        
        logger.info(f"Generating succinct final answer for query: '{query[:50]}...'")
        
        try:
            print("🔄 Processing final answer request...")
            messages = FINAL_ANSWER_PROMPT.format_messages(
                query=query,
                summarized_answer=summarized_answer
            )
            response = self.llm.invoke(messages)
            
            final_answer = response.content.strip()
            
            print(f"✅ FINAL ANSWER COMPLETE")
            print(f"🏆 FINAL ANSWER: '{final_answer}'")
            print("→→→ WORKFLOW COMPLETE!")
            
            logger.info(f"Final succinct answer: {final_answer}")
            
            return {
                **state,
                "final_answer": final_answer
            }
            
        except Exception as e:
            logger.error(f"Final answer generation error: {e}")
            error_msg = f"Error generating final answer: {str(e)}"
            print(f"❌ Final Answer Error: {str(e)}")
            print("→→→ WORKFLOW COMPLETE WITH ERROR!")
            return {
                **state,
                "final_answer": error_msg
            }
