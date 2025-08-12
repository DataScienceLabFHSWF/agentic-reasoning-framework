"""
retriever_agent.py
Agent for retrieving relevant documents from the knowledge base
"""

import logging
from typing import Dict, Any
import sys
import os

# Add your project path here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_utils.retriever import hybrid_retrieve
from .chat_state import ChatState

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Agent for document retrieval using hybrid search"""
    
    def __init__(self, chroma_dir: str, processed_dir: str, k: int = 3):
        self.chroma_dir = chroma_dir
        self.processed_dir = processed_dir
        self.k = k
        logger.info(f"Retriever agent initialized with k={k}")
        logger.info(f"ChromaDB path: {chroma_dir}")
        logger.info(f"Processed files path: {processed_dir}")
    
    def retrieve_documents(self, state: ChatState) -> Dict[str, Any]:
        """Retrieve relevant documents using hybrid retrieval with proper scoring"""
        query = state["query"]
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        
        try:
            retrieved_docs = hybrid_retrieve(
                query=query,
                chroma_dir=self.chroma_dir,
                processed_dir=self.processed_dir,
                k=self.k
            )
            
            doc_count = len(retrieved_docs)
            logger.info(f"Retrieved {doc_count} documents")
            
            # Extract and log relevance scores
            relevance_scores = []
            if doc_count > 0:
                logger.info("Document sources and relevance scores:")
                for i, doc in enumerate(retrieved_docs):
                    source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                    # Get combined relevance score from metadata
                    score = getattr(doc, 'metadata', {}).get('score', 0.0)
                    vector_score = getattr(doc, 'metadata', {}).get('vector_score', 0.0)
                    bm25_score = getattr(doc, 'metadata', {}).get('bm25_score', 0.0)
                    
                    relevance_scores.append(score)
                    logger.info(f"  Doc {i+1}: {source}")
                    logger.info(f"    Combined: {score:.3f} (Vector: {vector_score:.3f}, BM25: {bm25_score:.3f})")
                
                # Calculate max relevance score for routing decision
                max_score = max(relevance_scores) if relevance_scores else 0.0
                avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
                
                logger.info(f"Score summary - Max: {max_score:.3f}, Avg: {avg_score:.3f}")
            else:
                logger.warning("No documents retrieved")
                max_score = 0.0
            
            return {
                **state,
                "retrieved_docs": retrieved_docs,
                "max_relevance_score": max_score
            }
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return {
                **state,
                "retrieved_docs": [],
                "max_relevance_score": 0.0
            }