from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# --- Load BM25 retriever from markdown docs ---
def load_bm25_retriever(processed_dir: str, k: int = 5) -> BM25Retriever:
    loader = DirectoryLoader(processed_dir, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    return bm25

# --- Load Chroma retriever ---
def load_vector_retriever(chroma_dir: str, k: int = 5) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding_model)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

# --- Get vector similarity scores ---
def get_vector_scores(query: str, chroma_dir: str, k: int = 5) -> List[Tuple[Document, float]]:
    """Get documents with similarity scores from vector store"""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding_model)
    
    # Use similarity_search_with_score to get actual scores
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    # Convert distance to similarity (ChromaDB returns distance, lower is better)
    # Convert to similarity score (higher is better) using exponential decay
    scored_docs = []
    for doc, distance in docs_and_scores:
        # Convert distance to similarity score (0-1 range)
        similarity = np.exp(-distance)
        scored_docs.append((doc, similarity))
    
    return scored_docs

# --- Get BM25 scores ---
def get_bm25_scores(query: str, processed_dir: str, k: int = 5) -> List[Tuple[Document, float]]:
    """Get documents with BM25 scores"""
    loader = DirectoryLoader(processed_dir, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    
    # Get BM25 scores manually
    docs = bm25.get_relevant_documents(query)
    
    # Calculate BM25 scores for retrieved documents
    scored_docs = []
    for doc in docs:
        # Simple BM25-like scoring based on term frequency
        query_terms = query.lower().split()
        doc_text = doc.page_content.lower()
        
        score = 0.0
        for term in query_terms:
            # Simple term frequency scoring
            tf = doc_text.count(term)
            if tf > 0:
                score += tf / len(doc_text.split()) * 10  # Normalize and amplify
        
        scored_docs.append((doc, min(score, 1.0)))  # Cap at 1.0
    
    return scored_docs

# --- Combine scores from both retrievers ---
def combine_retrieval_scores(
    vector_docs: List[Tuple[Document, float]], 
    bm25_docs: List[Tuple[Document, float]],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> List[Tuple[Document, float]]:
    """Combine and deduplicate documents from both retrievers with weighted scores"""
    
    # Create a dict to store combined scores by content hash
    combined_scores = {}
    
    # Add vector scores
    for doc, score in vector_docs:
        doc_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
        combined_scores[doc_hash] = {
            'doc': doc,
            'vector_score': score * vector_weight,
            'bm25_score': 0.0
        }
    
    # Add BM25 scores
    for doc, score in bm25_docs:
        doc_hash = hash(doc.page_content[:100])
        if doc_hash in combined_scores:
            combined_scores[doc_hash]['bm25_score'] = score * bm25_weight
        else:
            combined_scores[doc_hash] = {
                'doc': doc,
                'vector_score': 0.0,
                'bm25_score': score * bm25_weight
            }
    
    # Calculate final scores and create result list
    final_docs = []
    for doc_hash, data in combined_scores.items():
        final_score = data['vector_score'] + data['bm25_score']
        # Add score to document metadata
        doc = data['doc']
        if not hasattr(doc, 'metadata') or doc.metadata is None:
            doc.metadata = {}
        doc.metadata['score'] = final_score
        doc.metadata['vector_score'] = data['vector_score']
        doc.metadata['bm25_score'] = data['bm25_score']
        
        final_docs.append((doc, final_score))
    
    # Sort by final score (descending)
    final_docs.sort(key=lambda x: x[1], reverse=True)
    
    return final_docs

# --- Main retrieval method with proper scoring ---
def hybrid_retrieve(query: str, chroma_dir: str, processed_dir: str, k: int = 5) -> List[Document]:
    """
    Hybrid retrieval with proper relevance scoring
    Returns documents with score, vector_score, and bm25_score in metadata
    """
    logger.info(f"Starting hybrid retrieval for query: '{query[:50]}...'")
    
    try:
        # Get scored documents from both retrievers
        logger.info("Retrieving from vector store...")
        vector_docs = get_vector_scores(query, chroma_dir, k)
        logger.info(f"Vector retriever found {len(vector_docs)} documents")
        
        logger.info("Retrieving from BM25...")
        bm25_docs = get_bm25_scores(query, processed_dir, k)
        logger.info(f"BM25 retriever found {len(bm25_docs)} documents")
        
        # Combine and score
        logger.info("Combining retrieval results...")
        combined_docs = combine_retrieval_scores(vector_docs, bm25_docs)
        
        # Return top k documents
        result_docs = [doc for doc, score in combined_docs[:k]]
        
        logger.info(f"Hybrid retrieval completed, returning {len(result_docs)} documents")
        for i, doc in enumerate(result_docs):
            score = doc.metadata.get('score', 0.0)
            logger.info(f"  Result {i+1}: score={score:.3f}")
        
        return result_docs
        
    except Exception as e:
        logger.error(f"Error in hybrid retrieval: {e}")
        return []

# --- Legacy function for backward compatibility ---
def load_hybrid_retriever(chroma_dir: str, processed_dir: str, k: int = 5) -> EnsembleRetriever:
    """Legacy function - kept for backward compatibility but not recommended"""
    logger.warning("Using legacy load_hybrid_retriever - scores will not be available")
    retriever_vector = load_vector_retriever(chroma_dir, k)
    retriever_bm25 = load_bm25_retriever(processed_dir, k)
    return EnsembleRetriever(retrievers=[retriever_vector, retriever_bm25], weights=[0.5, 0.5])