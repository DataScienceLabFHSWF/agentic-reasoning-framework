"""
Set of utils for agentrf operations

Includes:
- LRU-cached, document-scoped instances of AgenticRAGChat.
- Document ingestion utilities (File -> Markdown -> Vector DB).
"""

import os
from functools import lru_cache
from uuid import UUID
from pathlib import Path

try:
    from django.conf import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None  # Fallback for non-Django usage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Internal imports
# Adjust imports to match your project structure if 'agentrf' is not the root
from agent_utils.agentic_rag_chat import create_rag_chat
from data_utils.doc_processor import DocProcessor
from data_utils.chroma_db_from_md import load_hf_embeddings_from_env


@lru_cache(maxsize=20)
def get_rag_chat_for_document(document_id: UUID):
    """
    Get or create a document-scoped AgenticRAGChat instance.

    Each Document maintains its own conversation state within its cached instance.
    The instance is cached using LRU eviction (max 20 active documents).

    Args:
        document_id: UUID of the Document

    Returns:
        AgenticRAGChat instance configured from Django settings

    Raises:
        ValueError: if AGENTRF_CONFIG is not configured in settings
    """
    if not settings or not hasattr(settings, 'AGENTRF_CONFIG'):
        raise ValueError("AGENTRF_CONFIG not found in Django settings. Ensure it is configured in _django_base.py")

    config = settings.AGENTRF_CONFIG

    rag_chat = create_rag_chat(
        chroma_dir=config['chroma_dir'],
        processed_dir=config['processed_dir'],
        router_model=config['models']['router'],
        summarizer_model=config['models']['summarizer'],
        general_model=config['models']['general'],
        relevance_threshold=config['relevance_threshold'],
        max_react_iterations=config.get('max_react_iterations', 3),
        react_relevance_threshold=config.get('react_relevance_threshold', 0.1),
    )

    return rag_chat


def clear_document_chat_cache(document_id: UUID = None):
    """
    Clear the RAG chat cache for a specific document or all documents.

    Args:
        document_id: UUID of document to clear, or None to clear all

    Raises:
        NotImplementedError: if document_id is provided (selective eviction not supported)
    """
    if document_id is not None:
        raise NotImplementedError(
            "Selective cache eviction is not supported with functools.lru_cache. "
            "Use clear_document_chat_cache() without arguments to clear all, "
            "or consider migrating to cachetools.LRUCache for selective eviction."
        )

    # Clear entire cache
    get_rag_chat_for_document.cache_clear()


def ingest_file_to_chroma(file_path: str, persist_directory: str = None) -> bool:
    """
    End-to-end processing: Converts a raw file (PDF, Docx, etc.) to Markdown 
    and adds it to the ChromaDB vector store.

    Uses settings.AGENTRF_CONFIG['chroma_dir'] by default if persist_directory is not provided.

    Args:
        file_path (str): Path to the input file (e.g., "media/uploads/report.pdf").
        persist_directory (str, optional): Override the ChromaDB directory. 
                                           Defaults to Django settings if None.

    Returns:
        bool: True if successful, False otherwise.
    """
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        print(f"Error: File not found at {file_path}")
        return False

    # Resolve Persistence Directory from Settings if not provided
    if not persist_directory:
        if settings and hasattr(settings, 'AGENTRF_CONFIG'):
            persist_directory = settings.AGENTRF_CONFIG.get('chroma_dir')
        
        # Fallback if settings are missing (for standalone testing)
        if not persist_directory:
            persist_directory = "chroma_db"

    print(f"Starting ingestion for: {path_obj.name} -> {persist_directory}")

    # Step 1: Convert File to Markdown using DocProcessor
    try:
        processor = DocProcessor()
        # Dynamically find the correct handler for this file extension
        handler = processor._dispatch.get(path_obj.suffix.lower())
        
        if not handler:
            print(f"Error: Unsupported file type '{path_obj.suffix}'")
            print(f"Supported types: {', '.join(processor._dispatch.keys())}")
            return False
            
        # Extract text/markdown content using the handler
        md_content = handler(path_obj)
        
        if not md_content:
            print(f"Warning: Extracted content is empty for {path_obj.name}")
            return False
        
        print(f"Successfully extracted {len(md_content)} characters from {path_obj.name}")
            
    except Exception as e:
        print(f"Conversion Error: {e}")
        return False

    # Step 2: Prepare for Vector DB (Split & Embed)
    try:
        # Load embedding model
        embedding_model = load_hf_embeddings_from_env()
        
        # Configure text splitter (Must match settings in chroma_db_from_md.py)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Create base document
        doc = Document(
            page_content=md_content, 
            metadata={"source": str(path_obj.resolve())}
        )
        
        # Split into chunks
        chunks = text_splitter.split_documents([doc])
        
        if not chunks:
            print("Warning: No chunks created from document.")
            return False

        # Add metadata required for retrieval context
        base_filename = path_obj.name
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'filename': base_filename,
                'chunk_id': i,
                'total_chunks': len(chunks)
            })
        
        print(f"Created {len(chunks)} chunks from document")

    except Exception as e:
        print(f"Preparation Error: {e}")
        return False

    # Step 3: Add to ChromaDB
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        
        # Add documents to the store
        vector_store.add_documents(chunks)
        
        # Persist to disk (Required for older Chroma versions)
        if hasattr(vector_store, 'persist'):
            vector_store.persist()
            
        print(f"Successfully ingested {len(chunks)} chunks into ChromaDB at {persist_directory}")
        return True

    except Exception as e:
        print(f"Database Error: {e}")
        return False