from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from agentrf.doc_storage.chunks import load_chunks_jsonl
from agentrf.doc_storage.keyword import BM25Index

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Pure BM25 retriever backed by canonical chunks.

    Preferred startup path:
    - load saved BM25 artifact if present
    Fallback:
    - load canonical chunks and rebuild BM25
    """

    def __init__(
        self,
        chunks_path: str | Path,
        bm25_k: int = 5,
        bm25_index_path: Optional[str | Path] = None,
    ) -> None:
        self.chunks_path = Path(chunks_path).resolve()
        self.bm25_k = bm25_k
        self.bm25_index_path = Path(bm25_index_path).resolve() if bm25_index_path else None

        self.retriever: BM25Retriever = self._init_bm25()

    def _init_bm25(self) -> BM25Retriever:
        if self.bm25_index_path and self.bm25_index_path.exists():
            logger.info("Loading BM25 index from %s", self.bm25_index_path)
            index = BM25Index.load(self.bm25_index_path)
            index.retriever.k = self.bm25_k
            return index.retriever

        logger.info("BM25 artifact not found; rebuilding from canonical chunks at %s", self.chunks_path)
        chunks = load_chunks_jsonl(self.chunks_path)
        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = self.bm25_k
        return retriever

    @staticmethod
    def _ensure_chunk_metadata(doc: Document, default_chunk_id: int = 0) -> Document:
        if doc.metadata is None:
            doc.metadata = {}

        if "chunk_id" not in doc.metadata:
            doc.metadata["chunk_id"] = default_chunk_id
            doc.metadata["chunk_id_inferred"] = True

        src = doc.metadata.get("source", "")
        if src and "filename" not in doc.metadata:
            doc.metadata["filename"] = Path(src).name

        return doc

    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        results: List[Document] = []
        seen_content: set[str] = set()

        try:
            docs = self.retriever.invoke(query)
            for doc in docs:
                if doc.page_content in seen_content:
                    continue

                results.append(self._ensure_chunk_metadata(doc))
                seen_content.add(doc.page_content)

        except Exception as e:
            logger.error("BM25 retrieval failed: %s", e)

        return results[: (top_k or self.bm25_k)]
