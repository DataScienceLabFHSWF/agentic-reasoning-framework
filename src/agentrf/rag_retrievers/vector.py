# Vector Retriever using ChromaDB for persistent vector storage and retrieval.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Pure vector retriever backed by persistent ChromaDB.
    """

    def __init__(
        self,
        chroma_persist_dir: str | Path,
        embedding_function: Embeddings,
        vector_k: int = 5,
    ) -> None:
        self.chroma_dir = Path(chroma_persist_dir).resolve()
        self.embedding_function = embedding_function
        self.vector_k = vector_k

        logger.info("Initializing Chroma Vector Store from %s", self.chroma_dir)

        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_dir),
            embedding_function=self.embedding_function,
        )
        self.vector_retriever: VectorStoreRetriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.vector_k}
        )

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
            docs = self.vector_retriever.invoke(query)
            for doc in docs:
                if doc.page_content in seen_content:
                    continue

                results.append(self._ensure_chunk_metadata(doc))
                seen_content.add(doc.page_content)

        except Exception as e:
            logger.error("Vector retrieval failed: %s", e)

        return results[: (top_k or self.vector_k)]