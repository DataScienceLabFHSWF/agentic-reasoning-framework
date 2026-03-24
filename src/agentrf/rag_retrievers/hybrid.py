from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from .bm25 import BM25Retriever
from .vector import VectorRetriever


class HybridRetriever:
    """
    Compose separate vector and BM25 retrievers.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        results: List[Document] = []
        seen_content: set[str] = set()

        for doc in self.vector_retriever.retrieve(query, top_k=top_k):
            if doc.page_content not in seen_content:
                results.append(doc)
                seen_content.add(doc.page_content)

        for doc in self.bm25_retriever.retrieve(query, top_k=top_k):
            if doc.page_content not in seen_content:
                results.append(doc)
                seen_content.add(doc.page_content)

        return results[:top_k]
