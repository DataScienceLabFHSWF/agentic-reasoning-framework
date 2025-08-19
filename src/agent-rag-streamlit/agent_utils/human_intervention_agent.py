"""
human_intervention_agent.py
Agent for handling human intervention in the workflow
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import HUMAN_INTERVENTION_PROMPT

logger = logging.getLogger(__name__)


class HumanInterventionAgent:
    """Agent for handling human intervention commands"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("Human intervention agent initialized")
    
    def handle_intervention(self, state: ChatState) -> Dict[str, Any]:
        """Handle human intervention commands and route accordingly"""
        query = state["query"]
        command = state.get("human_intervention", "")
        
        logger.info(f"Processing human intervention: {command}")
        
        try:
            # Explain the intervention to the user
            messages = HUMAN_INTERVENTION_PROMPT.format_messages(
                command=command,
                query=query
            )
            explanation = self.llm.invoke(messages).content
            
            # Set routing flags based on command
            if command == "force_rag":
                logger.info("Human intervention: Forcing RAG search")
                return {
                    **state,
                    "human_intervention_explanation": explanation,
                    "force_rag": True,
                    "is_relevant": True  # Override relevance check
                }
            elif command == "use_context":
                logger.info("Human intervention: Using only context")
                return {
                    **state,
                    "human_intervention_explanation": explanation,
                    "use_context_only": True
                }
            elif command == "general":
                logger.info("Human intervention: Treating as general question")
                return {
                    **state,
                    "human_intervention_explanation": explanation,
                    "force_general": True
                }
            else:
                logger.warning(f"Unknown intervention command: {command}")
                return {
                    **state,
                    "human_intervention_explanation": f"Unbekannter Befehl: {command}",
                    "force_general": True
                }
                
        except Exception as e:
            logger.error(f"Human intervention error: {e}")
            return {
                **state,
                "human_intervention_explanation": f"Fehler bei der Verarbeitung des Befehls: {str(e)}",
                "force_general": True
            }
    
    def route_decision(self, state: ChatState) -> str:
        """Decision function for routing after human intervention"""
        if state.get("force_rag", False):
            return "force_rag"
        elif state.get("use_context_only", False):
            return "use_context"
        else:
            return "general"
