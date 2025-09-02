"""
reasoning_agent.py
Agent for deep reasoning over retrieved documents
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import REASONING_PROMPT

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Agent for performing deep reasoning over retrieved documents"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("Reasoning agent initialized")
    
    def reason_over_documents(self, state: ChatState) -> Dict[str, Any]:
        """Perform deep reasoning analysis over retrieved documents"""
        query = state["query"]
        retrieved_docs = state.get("retrieved_docs", [])
        
        print("\n🧠 REASONING AGENT")
        print("-" * 50)
        print(f"📖 Analyzing {len(retrieved_docs)} documents")
        
        logger.info(f"Reasoning over {len(retrieved_docs)} documents for query: '{query[:50]}...'")
        
        if not retrieved_docs:
            reasoning_answer = "Keine Dokumente verfügbar für die Analyse."
            print("❌ No documents to analyze")
            print("→→→ Proceeding with empty reasoning...")
        else:
            try:
                # Format documents for reasoning
                context = ""
                for i, doc in enumerate(retrieved_docs, 1):
                    source = getattr(doc, 'metadata', {}).get('source', f'Dokument {i}')
                    content = doc.page_content
                    context += f"Document {i} ({source}):\n{content}\n\n"
                
                print(f"📝 Context Length: {len(context)} characters")
                
                messages = REASONING_PROMPT.format_messages(
                    query=query,
                    context=context
                )
                
                print("🔄 Processing reasoning request...")
                response = self.llm.invoke(messages)
                reasoning_answer = response.content
                
                print(f"✅ REASONING COMPLETE")
                print(f"📏 Answer Length: {len(reasoning_answer)} characters")
                print(f"🎯 Answer Preview: {reasoning_answer[:200]}...")
                print("→→→ Proceeding to SUMMARIZER AGENT...")
                
                logger.info(f"Reasoning completed successfully: {len(reasoning_answer)} chars")
                
            except Exception as e:
                logger.error(f"Reasoning error: {e}")
                reasoning_answer = f"Fehler beim Reasoning: {str(e)}"
                print(f"❌ Reasoning Error: {str(e)}")
                print("→→→ Proceeding with error message...")
        
        return {
            **state,
            "reasoning_answer": reasoning_answer
        }