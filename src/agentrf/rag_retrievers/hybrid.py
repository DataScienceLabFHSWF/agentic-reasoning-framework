import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retriever combining vector search (ChromaDB) and keyword search (BM25).
    Designed as a pure framework component: requires all dependencies to be injected.
    """
    def __init__(
        self,
        chroma_persist_dir: str | Path,
        processed_kb_dir: str | Path,
        embedding_function: Embeddings,
        vector_k: int = 5,
        bm25_k: int = 5,
    ):
        self.chroma_dir = Path(chroma_persist_dir).resolve()
        self.processed_dir = Path(processed_kb_dir).resolve()
        self.embedding_function = embedding_function
        self.vector_k = vector_k
        self.bm25_k = bm25_k

        # 1. Initialize Vector Store (Chroma)
        logger.info("Initializing Chroma Vector Store from %s", self.chroma_dir)
        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_dir),
            embedding_function=self.embedding_function
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.vector_k})

        # 2. Initialize Keyword Store (BM25)
        logger.info("Initializing BM25 Retriever from %s", self.processed_dir)
        self.bm25_retriever: Optional[BM25Retriever] = self._init_bm25()

    def _init_bm25(self) -> Optional[BM25Retriever]:
        """Loads markdown files dynamically to build the BM25 index in memory."""
        if not self.processed_dir.exists():
            logger.warning("Processed KB directory not found: %s", self.processed_dir)
            return None

        # Load all markdown files for keyword search
        loader = DirectoryLoader(
            str(self.processed_dir), 
            glob="**/*.md", 
            loader_cls=UnstructuredMarkdownLoader
        )
        docs = loader.load()
        
        if not docs:
            logger.warning("No markdown files found in %s for BM25.", self.processed_dir)
            return None

        retriever = BM25Retriever.from_documents(docs)
        retriever.k = self.bm25_k
        return retriever

    def _ensure_chunk_metadata(self, doc: Document, default_chunk_id: int = 0) -> Document:
        """Ensure document has chunk metadata. Add defaults if missing."""
        if not hasattr(doc, 'metadata') or doc.metadata is None:
            doc.metadata = {}
        
        # If chunk_id is missing, try to infer or set default
        if 'chunk_id' not in doc.metadata:
            doc.metadata['chunk_id'] = default_chunk_id
            doc.metadata['chunk_id_inferred'] = True
        
        # Ensure filename is set
        src = doc.metadata.get('source', '')
        if src and 'filename' not in doc.metadata:
            doc.metadata['filename'] = Path(src).name
            
        return doc

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Executes both Vector and BM25 search, deduplicates the results, 
        ensures metadata, and returns the top combined documents.
        """
        results: List[Document] = []
        seen_content = set()

        # 1. Execute Vector Search
        logger.debug("Executing Vector Search for: '%s'", query)
        try:
            vec_docs = self.vector_retriever.invoke(query)
            for doc in vec_docs:
                if doc.page_content not in seen_content:
                    doc = self._ensure_chunk_metadata(doc)
                    results.append(doc)
                    seen_content.add(doc.page_content)
        except Exception as e:
            logger.error("Vector retrieval failed: %s", e)

        # 2. Execute BM25 Search
        if self.bm25_retriever:
            logger.debug("Executing BM25 Search for: '%s'", query)
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
                for doc in bm25_docs:
                    if doc.page_content not in seen_content:
                        doc = self._ensure_chunk_metadata(doc)
                        results.append(doc)
                        seen_content.add(doc.page_content)
            except Exception as e:
                logger.error("BM25 retrieval failed: %s", e)

        # Truncate to the requested top_k
        return results[:top_k]