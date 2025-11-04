# retriever.py  (enhanced with chunk-level retrieval and context expansion)

from __future__ import annotations
import logging
import hashlib
from typing import List, Tuple, Dict, Any, Optional
import os 
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior"""
    k: int = 3
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    enable_mmr: bool = False
    mmr_diversity_threshold: float = 0.8
    context_window: int = 0  # 0 = no expansion, >0 = chunks before/after
    merge_strategy: str = "separate"  # "separate", "merge", "hierarchical"

# ---------- Utilities

def _stable_chunk_key(doc: Document) -> str:
    """
    Build a stable, collision-resistant key for a chunk-level Document.
    Uses source file path + chunk_id for stable identification.
    Falls back to content hash if chunk_id is not available.
    """
    src = str(doc.metadata.get("source", ""))
    chunk_id = doc.metadata.get("chunk_id")
    
    if src and chunk_id is not None:
        return f"{src}::chunk_{chunk_id}"
    elif src:
        # For legacy documents without chunk_id, use source + content hash
        content_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:8]
        return f"{src}::legacy_{content_hash}"
    else:
        # Last resort: content hash only
        return hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()

def _normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize to [0,1] (robust to constant inputs)."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo + 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def apply_mmr_diversification(
    docs_scores: List[Tuple[Document, float]], 
    diversity_threshold: float = 0.8,
    final_k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Apply Maximal Marginal Relevance to diversify results.
    Simple implementation based on content similarity.
    """
    if len(docs_scores) <= final_k:
        return docs_scores
    
    selected = []
    remaining = docs_scores.copy()
    
    # Start with highest scored document
    selected.append(remaining.pop(0))
    
    while len(selected) < final_k and remaining:
        best_idx = 0
        best_score = float('-inf')
        
        for i, (candidate_doc, relevance_score) in enumerate(remaining):
            # Calculate diversity penalty (simple word overlap)
            max_similarity = 0.0
            candidate_words = set(candidate_doc.page_content.lower().split())
            
            for selected_doc, _ in selected:
                selected_words = set(selected_doc.page_content.lower().split())
                if candidate_words and selected_words:
                    overlap = len(candidate_words & selected_words)
                    similarity = overlap / min(len(candidate_words), len(selected_words))
                    max_similarity = max(max_similarity, similarity)
            
            # MMR score: relevance - λ * max_similarity
            mmr_score = relevance_score - diversity_threshold * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        selected.append(remaining.pop(best_idx))
    
    return selected

# ---------- Vector retrieval with robust similarity

def get_vector_scores(query: str, chroma_dir: str, k: int = 5) -> List[Tuple[Document, float]]:
    """
    Retrieve K docs from Chroma with a robust similarity mapping.
    Handles both legacy documents (without chunk_id) and new documents (with chunk_id).
    """
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding_model)

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

    scored_docs: List[Tuple[Document, float]] = []
    for i, (doc, distance) in enumerate(docs_and_scores):
        d = float(distance)
        if 0.0 <= d <= 1.5:
            sim = max(0.0, 1.0 - d)  # cosine-like
        else:
            sim = 1.0 / (1.0 + d)    # generic
        
        # Ensure metadata consistency
        doc = _ensure_chunk_metadata(doc, default_chunk_id=i)
        scored_docs.append((doc, sim))
    return scored_docs

# ---------- BM25 retrieval with chunk-level support

def get_bm25_scores_chunked(query: str, processed_dir: str, k: int = 5) -> List[Tuple[Document, float]]:
    """
    BM25 retrieval that maintains chunk-level granularity.
    """
    # Load and chunk documents the same way as during ingestion
    markdown_files = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    
    if not markdown_files:
        return []

    all_chunks = []
    
    for file_path in markdown_files:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            for doc in docs:
                chunks = text_splitter.split_documents([doc])
                
                for i, chunk in enumerate(chunks):
                    if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata['source'] = file_path
                    chunk.metadata['filename'] = os.path.basename(file_path)
                    chunk.metadata['chunk_id'] = i
                    chunk.metadata['total_chunks'] = len(chunks)
                    all_chunks.append(chunk)
                    
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue

    if not all_chunks:
        return []

    # Try real BM25 on chunks
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        
        def tok(text: str) -> List[str]:
            return [t for t in text.lower().split() if t]

        corpus_tokens = [tok(chunk.page_content) for chunk in all_chunks]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = tok(query)
        scores = bm25.get_scores(query_tokens)

        # Take top-k by score
        indices = np.argsort(scores)[::-1][:k]
        top = []
        norm = _normalize_scores([float(scores[i]) for i in indices])
        for idx, s in zip(indices, norm):
            top.append((all_chunks[int(idx)], float(s)))
        return top

    except ImportError:
        logger.warning("rank_bm25 not available, using fallback BM25 implementation")
        # Fallback: pseudo-BM25 using simple TF frequency with IDF proxy
        from collections import Counter
        corpus_terms = [set(d.page_content.lower().split()) for d in all_chunks]
        df = Counter()
        for terms in corpus_terms:
            df.update(terms)
        N = len(all_chunks)

        def idf(term: str) -> float:
            return np.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)

        q_terms = [t for t in query.lower().split() if t]
        raw_scores: List[float] = []
        for d in all_chunks:
            text = d.page_content.lower()
            words = text.split()
            L = max(1, len(words))
            s = 0.0
            for t in q_terms:
                tf = text.count(t)
                if tf:
                    s += (tf / L) * idf(t)
            raw_scores.append(s)

        # Select top-k and normalize
        idxs = np.argsort(raw_scores)[::-1][:k]
        norm = _normalize_scores([raw_scores[i] for i in idxs])
        return [(all_chunks[int(i)], float(s)) for i, s in zip(idxs, norm)]
    except Exception as e:
        logger.error(f"BM25 scoring failed: {e}")
        return []

# ---------- Context expansion functions

def expand_chunk_context(
    chunks: List[Document], 
    processed_dir: str, 
    window_size: int = 1
) -> List[Document]:
    """
    Expand retrieved chunks with neighboring context.
    Handles both legacy and chunk-aware documents.
    """
    if window_size <= 0:
        return chunks
    
    expanded_chunks = []
    processed_keys = set()
    
    for target_chunk in chunks:
        source_file = target_chunk.metadata.get('source', '')
        target_chunk_id = target_chunk.metadata.get('chunk_id', 0)
        is_chunk_id_inferred = target_chunk.metadata.get('chunk_id_inferred', False)
        
        if not source_file:
            expanded_chunks.append(target_chunk)
            continue
        
        # If chunk_id was inferred (legacy document), skip expansion
        if is_chunk_id_inferred:
            logger.warning(f"Skipping context expansion for legacy document without stable chunk_id: {source_file}")
            if _stable_chunk_key(target_chunk) not in processed_keys:
                expanded_chunks.append(target_chunk)
                processed_keys.add(_stable_chunk_key(target_chunk))
            continue
        
        try:
            # Get neighbors including the target chunk itself
            neighbors = get_chunk_neighbors(source_file, target_chunk_id, processed_dir, window_size)
            
            for neighbor in neighbors:
                neighbor_key = _stable_chunk_key(neighbor)
                if neighbor_key not in processed_keys:
                    # Mark context expansion metadata
                    neighbor_chunk_id = neighbor.metadata.get('chunk_id', 0)
                    neighbor.metadata['is_expanded_context'] = (neighbor_chunk_id != target_chunk_id)
                    neighbor.metadata['target_chunk_id'] = target_chunk_id
                    
                    # Copy over retrieval scores from target chunk if this is a context chunk
                    if neighbor_chunk_id != target_chunk_id:
                        neighbor.metadata['score'] = target_chunk.metadata.get('score', 0.0)
                        neighbor.metadata['vector_score'] = target_chunk.metadata.get('vector_score', 0.0)
                        neighbor.metadata['bm25_score'] = target_chunk.metadata.get('bm25_score', 0.0)
                    
                    expanded_chunks.append(neighbor)
                    processed_keys.add(neighbor_key)
                    
        except Exception as e:
            logger.warning(f"Failed to expand context for chunk {target_chunk_id} from {source_file}: {e}")
            if _stable_chunk_key(target_chunk) not in processed_keys:
                expanded_chunks.append(target_chunk)
                processed_keys.add(_stable_chunk_key(target_chunk))
    
    return expanded_chunks

def get_chunk_neighbors(source_file: str, chunk_id: int, processed_dir: str, window_size: int = 1) -> List[Document]:
    """
    Retrieve neighboring chunks around a specific chunk for context expansion.
    """
    try:
        loader = UnstructuredMarkdownLoader(source_file)
        docs = loader.load()
        
        if not docs:
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(docs)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['source'] = source_file
            chunk.metadata['filename'] = os.path.basename(source_file)
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        # Extract window around target chunk
        start_idx = max(0, chunk_id - window_size)
        end_idx = min(len(chunks), chunk_id + window_size + 1)
        
        return chunks[start_idx:end_idx]
        
    except Exception as e:
        logger.error(f"Error retrieving neighbors for {source_file}, chunk {chunk_id}: {e}")
        return []

def merge_consecutive_chunks(chunks: List[Document]) -> List[Document]:
    """
    Merge consecutive chunks from the same document into larger passages.
    Only merges if chunks are actually consecutive.
    """
    if not chunks:
        return []
    
    # Group by source file
    by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get('source', '')
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(chunk)
    
    merged_docs = []
    
    for source, source_chunks in by_source.items():
        # Sort by chunk_id
        source_chunks.sort(key=lambda x: x.metadata.get('chunk_id', 0))
        
        # Group consecutive chunks
        current_group = [source_chunks[0]]
        
        for chunk in source_chunks[1:]:
            prev_chunk_id = current_group[-1].metadata.get('chunk_id', 0)
            curr_chunk_id = chunk.metadata.get('chunk_id', 0)
            
            if curr_chunk_id == prev_chunk_id + 1:
                current_group.append(chunk)
            else:
                # Merge current group if it has multiple chunks, otherwise keep individual
                if len(current_group) > 1:
                    merged_doc = _merge_chunk_group(current_group)
                    merged_docs.append(merged_doc)
                else:
                    merged_docs.extend(current_group)
                current_group = [chunk]
        
        # Handle final group
        if len(current_group) > 1:
            merged_doc = _merge_chunk_group(current_group)
            merged_docs.append(merged_doc)
        else:
            merged_docs.extend(current_group)
    
    return merged_docs

def _merge_chunk_group(chunks: List[Document]) -> Document:
    """Merge a group of consecutive chunks into a single document."""
    if not chunks:
        return Document(page_content="", metadata={})
    
    # Combine content
    combined_content = "\n\n".join(chunk.page_content for chunk in chunks)
    
    # Combine metadata
    first_chunk = chunks[0]
    last_chunk = chunks[-1]
    
    merged_metadata = first_chunk.metadata.copy()
    merged_metadata['chunk_id_start'] = first_chunk.metadata.get('chunk_id', 0)
    merged_metadata['chunk_id_end'] = last_chunk.metadata.get('chunk_id', 0)
    merged_metadata['merged_chunk_count'] = len(chunks)
    merged_metadata['is_merged'] = True
    
    # Combine scores if present
    total_score = sum(float(chunk.metadata.get('score', 0.0)) for chunk in chunks)
    merged_metadata['score'] = total_score / len(chunks)  # Average score
    
    return Document(page_content=combined_content, metadata=merged_metadata)

# ---------- Enhanced combination function

def combine_retrieval_scores(
    vector_docs: List[Tuple[Document, float]],
    bm25_docs: List[Tuple[Document, float]],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Tuple[Document, float]]:
    """
    Combine and deduplicate documents from both retrievers with weighted scores.
    Deduplication uses chunk-level stable keys (source + chunk_id).
    """
    combined: Dict[str, Dict[str, Any]] = {}

    # vector
    for doc, vscore in vector_docs:
        key = _stable_chunk_key(doc)
        combined[key] = {
            "doc": doc,
            "vector_score": float(vscore) * float(vector_weight),
            "bm25_score": 0.0,
        }

    # bm25
    for doc, bscore in bm25_docs:
        key = _stable_chunk_key(doc)
        if key in combined:
            combined[key]["bm25_score"] = float(bscore) * float(bm25_weight)
        else:
            combined[key] = {
                "doc": doc,
                "vector_score": 0.0,
                "bm25_score": float(bscore) * float(bm25_weight),
            }

    # sum & sort
    final_list: List[Tuple[Document, float]] = []
    for data in combined.values():
        total = float(data["vector_score"]) + float(data["bm25_score"])
        doc = data["doc"]
        if not hasattr(doc, "metadata") or doc.metadata is None:
            doc.metadata = {}
        doc.metadata["score"] = total
        doc.metadata["vector_score"] = float(data["vector_score"])
        doc.metadata["bm25_score"] = float(data["bm25_score"])
        final_list.append((doc, total))

    final_list.sort(key=lambda x: x[1], reverse=True)
    return final_list

# ---------- Enhanced public hybrid entrypoint

def hybrid_retrieve(
    query: str,
    chroma_dir: str,
    processed_dir: str,
    k: int = 5,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    config: Optional[RetrievalConfig] = None,
) -> List[Document]:
    """
    Enhanced hybrid retrieval with chunk-level granularity and context expansion.
    Returns top-k Documents with scores in .metadata.
    """
    if config is None:
        config = RetrievalConfig(k=k, vector_weight=vector_weight, bm25_weight=bm25_weight)
    
    logger.info(f"Hybrid retrieval: '{query[:80]}'  (k={config.k}, vw={config.vector_weight}, bw={config.bm25_weight})")

    try:
        # Get vector and BM25 scores
        v_docs = get_vector_scores(query, chroma_dir, k=config.k * 2)
        b_docs = get_bm25_scores_chunked(query, processed_dir, k=config.k * 2)

        # Check if we have a mix of legacy and chunk-aware documents
        has_legacy_docs = any(doc.metadata.get('chunk_id_inferred', False) for doc, _ in v_docs)
        if has_legacy_docs:
            logger.warning("Detected legacy documents without stable chunk IDs. Consider regenerating ChromaDB with the updated ingestion script for full chunk-level functionality.")

        # Combine scores
        combined = combine_retrieval_scores(
            v_docs, b_docs, 
            vector_weight=config.vector_weight, 
            bm25_weight=config.bm25_weight
        )

        # Apply MMR if enabled
        if config.enable_mmr:
            combined = apply_mmr_diversification(
                combined, 
                diversity_threshold=config.mmr_diversity_threshold,
                final_k=config.k
            )
        
        # Get top chunks
        top_docs = [doc for doc, _ in combined[:config.k]]

        # Apply context expansion if requested (will be skipped for legacy docs)
        if config.context_window > 0:
            top_docs = expand_chunk_context(top_docs, processed_dir, config.context_window)
        
        # Apply merging strategy (will be limited for legacy docs)
        if config.merge_strategy == "merge":
            top_docs = merge_consecutive_chunks(top_docs)
        elif config.merge_strategy == "hierarchical":
            original_docs = top_docs.copy()
            merged_docs = merge_consecutive_chunks(top_docs.copy())
            
            final_docs = []
            
            # Add hierarchy metadata to originals
            for doc in original_docs:
                doc.metadata['hierarchy_level'] = 'chunk'
                final_docs.append(doc)
            
            # Only add merged versions if merging actually occurred
            for doc in merged_docs:
                if doc.metadata.get('is_merged', False):
                    doc.metadata['hierarchy_level'] = 'section'
                    final_docs.append(doc)
            
            top_docs = final_docs

        # Log results with legacy indicators
        for i, doc in enumerate(top_docs[:config.k], 1):
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            
            # Ensure filename is set
            src = doc.metadata.get("source", "")
            if src:
                doc.metadata["filename"] = os.path.basename(src)
            
            s = float(doc.metadata.get("score", 0.0))
            vs = float(doc.metadata.get("vector_score", 0.0))
            bs = float(doc.metadata.get("bm25_score", 0.0))
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            is_expanded = doc.metadata.get("is_expanded_context", False)
            target_chunk = doc.metadata.get("target_chunk_id", chunk_id)
            is_legacy = doc.metadata.get("chunk_id_inferred", False)
            
            context_marker = " [CONTEXT]" if is_expanded else ""
            legacy_marker = " [LEGACY]" if is_legacy else ""
            
            logger.info(f"#{i}: total={s:.4f} (v={vs:.4f}, bm25={bs:.4f}) chunk={chunk_id} target={target_chunk}{context_marker}{legacy_marker} src={doc.metadata.get('filename','')}")
        
        return top_docs

    except Exception as e:
        logger.exception(f"Hybrid retrieval failed: {e}")
        return []

# Convenience functions for specific retrieval strategies
def retrieve_with_expansion(
    query: str, chroma_dir: str, processed_dir: str, 
    k: int = 5, window_size: int = 1, **kwargs
) -> List[Document]:
    """Retrieve with automatic context window expansion."""
    config = RetrievalConfig(k=k, context_window=window_size, **kwargs)
    return hybrid_retrieve(query, chroma_dir, processed_dir, config=config)

def retrieve_with_mmr(
    query: str, chroma_dir: str, processed_dir: str,
    k: int = 5, diversity_threshold: float = 0.8, **kwargs
) -> List[Document]:
    """Retrieve with MMR diversification."""
    config = RetrievalConfig(k=k, enable_mmr=True, mmr_diversity_threshold=diversity_threshold, **kwargs)
    return hybrid_retrieve(query, chroma_dir, processed_dir, config=config)

def retrieve_hierarchical(
    query: str, chroma_dir: str, processed_dir: str,
    k: int = 5, **kwargs
) -> List[Document]:
    """Retrieve with hierarchical chunk + section results."""
    config = RetrievalConfig(k=k, merge_strategy="hierarchical", **kwargs)
    return hybrid_retrieve(query, chroma_dir, processed_dir, config=config)

def _ensure_chunk_metadata(doc: Document, default_chunk_id: int = 0) -> Document:
    """
    Ensure document has chunk metadata. Add defaults if missing.
    """
    if not hasattr(doc, 'metadata') or doc.metadata is None:
        doc.metadata = {}
    
    # If chunk_id is missing, try to infer or set default
    if 'chunk_id' not in doc.metadata:
        doc.metadata['chunk_id'] = default_chunk_id
        doc.metadata['chunk_id_inferred'] = True  # Mark as inferred
    
    # Ensure filename is set
    src = doc.metadata.get('source', '')
    if src and 'filename' not in doc.metadata:
        doc.metadata['filename'] = os.path.basename(src)
    
    return doc

def check_chromadb_chunk_compatibility(chroma_dir: str) -> bool:
    """
    Check if the ChromaDB contains documents with chunk IDs.
    Returns True if chunk-aware, False if legacy.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding_model)
        
        # Sample a few documents
        sample_docs = vectorstore.similarity_search("test", k=3)
        
        chunk_aware_count = 0
        for doc in sample_docs:
            if doc.metadata and 'chunk_id' in doc.metadata:
                chunk_aware_count += 1
        
        return chunk_aware_count > 0
        
    except Exception as e:
        logger.error(f"Failed to check ChromaDB compatibility: {e}")
        return False
