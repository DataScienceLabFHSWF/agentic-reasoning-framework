from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from agentrf.rag_retrievers import HybridRetriever, VectorRetriever
from agentrf.rag_retrievers.bm25 import BM25Retriever


class RetrieverFactory:
    @staticmethod
    def create(
        retriever_type: str,
        top_k: int,
        chroma_persist_dir: str | Path,
        chunks_path: str | Path,
        embedding_function: Embeddings,
        bm25_index_path: str | Path | None = None,
    ):
        retriever_name = retriever_type.lower()

        if retriever_name == "hybrid":
            vector_retriever = VectorRetriever(
                chroma_persist_dir=chroma_persist_dir,
                embedding_function=embedding_function,
                vector_k=top_k,
            )
            bm25_retriever = BM25Retriever(
                chunks_path=chunks_path,
                bm25_k=top_k,
                bm25_index_path=bm25_index_path,
            )
            return HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
            )

        if retriever_name == "vector":
            return VectorRetriever(
                chroma_persist_dir=chroma_persist_dir,
                embedding_function=embedding_function,
                vector_k=top_k,
            )

        if retriever_name == "bm25":
            return BM25Retriever(
                chunks_path=chunks_path,
                bm25_k=top_k,
                bm25_index_path=bm25_index_path,
            )

        raise ValueError(f"Unsupported retriever type: {retriever_name}")