"""
summarizer_agent.py
Agent for summarizing retrieved documents to answer queries
"""

import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .chat_state import ChatState
from .prompts import SUMMARIZER_PROMPT

logger = logging.getLogger(__name__)


class SummarizerAgent:
    """Agent for generating responses based on retrieved documents"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.info("Summarizer agent initialized")
    
    def summarize_response(self, state: ChatState) -> Dict[str, Any]:
        """Summarize the retrieved documents to answer the query"""
        query = state["query"]
        docs = state["retrieved_docs"]
        
        logger.info(f"Summarizing response for query: '{query[:50]}...'")
        logger.info(f"Processing {len(docs)} documents")
        
        if not docs:
            logger.warning("No documents available for summarization")
            answer = "No relevant documents were found in the knowledge base for your query."
        else:
            try:
                # Combine document contents
                context = "\n\n".join([
                    f"Document {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(docs)
                ])
                
                logger.info(f"Context length: {len(context)} characters")
                
                messages = SUMMARIZER_PROMPT.format_messages(query=query, context=context)
                response = self.llm.invoke(messages)
                answer = response.content
                
                logger.info("Summarization completed successfully")
                
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                answer = f"An error occurred while processing the documents: {str(e)}"
        
        return {
            **state,
            "answer": answer
        }