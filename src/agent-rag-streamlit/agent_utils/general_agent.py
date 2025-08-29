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
        """Handle queries where intent wasn't corpus-relevant or documents weren't relevant enough"""
        query = state["query"]
        max_score = state.get("max_relevance_score", 0.0)
        is_corpus_relevant = state.get("is_corpus_relevant", True)
        intent_reasoning = state.get("intent_reasoning", "")
        
        logger.info(f"Generating general response for: '{query[:50]}...'")
        
        if not is_corpus_relevant:
            logger.info("Query not relevant to nuclear corpus - providing general guidance")
            context = f"Die Anfrage wurde als nicht relevant für unsere Nuklear-Wissensdatenbank eingestuft. Grund: {intent_reasoning}"
        else:
            logger.info(f"Documents insufficient relevance (score: {max_score:.3f}) - suggesting rephrase")
            context = f"Dokumente hatten einen maximalen Relevanzwert von {max_score:.3f}, der unter dem Schwellenwert lag. Bitte formulieren Sie die Frage spezifischer."
        
        try:
            messages = GENERAL_PROMPT.format_messages(query=query, context=context)
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Ensure German-only response
            if not self._is_german_response(answer):
                logger.warning("Response not in German, requesting German translation")
                answer = self._ensure_german_response(answer, query)
            
            logger.info("General response generated successfully")
            
            return {
                **state,
                "summarized_answer": answer,
                "retrieved_docs": []
            }
            
        except Exception as e:
            logger.error(f"General response error: {e}")
            return {
                **state,
                "summarized_answer": f"Es trat ein Fehler bei der Verarbeitung Ihrer Anfrage auf: {str(e)}",
                "retrieved_docs": []
            }
    
    def _is_german_response(self, text: str) -> bool:
        """Check if response is primarily in German"""
        german_indicators = ['der', 'die', 'das', 'und', 'oder', 'aber', 'ich', 'Sie', 'wir', 'sind', 'haben', 'können']
        words = text.lower().split()
        if len(words) < 5:
            return True  # Too short to determine
        german_count = sum(1 for word in words if any(indicator in word for indicator in german_indicators))
        return german_count / len(words) > 0.3

    def _ensure_german_response(self, english_response: str, query: str) -> str:
        """Force German response if initial response wasn't in German"""
        try:
            german_prompt = f"""
            Übersetze die folgende Antwort ins Deutsche und stelle sicher, dass sie natürlich klingt:
            
            Ursprüngliche Antwort: {english_response}
            Ursprüngliche Frage: {query}
            
            Deutsche Antwort:
            """
            response = self.llm.invoke([{"role": "user", "content": german_prompt}])
            return response.content
        except Exception as e:
            logger.error(f"German translation failed: {e}")
            return f"Entschuldigung, es gab einen Fehler bei der Antwort auf Ihre Frage: {query}"