from __future__ import annotations

from langchain_core.embeddings import Embeddings

from agentrf.rag_retrievers import HybridRetriever, VectorRetriever
from agentrf.rag_retrievers.bm25 import BM25Retriever
from agentrf.settings import Settings


class RetrieverFactory:
    @staticmethod
    def create(
        settings: Settings,
        embedding_function: Embeddings,
        retriever_type: str | None = None,
    ):
        retriever_name = (retriever_type or settings.rag.retriever.type).lower()
        top_k = settings.rag.retriever.top_k

        if retriever_name == "hybrid":
            vector_retriever = VectorRetriever(
                chroma_persist_dir=settings.paths.chroma_db_dir,
                embedding_function=embedding_function,
                vector_k=top_k,
            )
            bm25_retriever = BM25Retriever(
                chunks_path=settings.paths.knowledge_base_processed,
                bm25_k=top_k,
                bm25_index_path=getattr(settings.paths, "bm25_index_path", None),
            )
            return HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
            )

        if retriever_name == "vector":
            return VectorRetriever(
                chroma_persist_dir=settings.paths.chroma_db_dir,
                embedding_function=embedding_function,
                vector_k=top_k,
            )

        if retriever_name == "bm25":
            return BM25Retriever(
                chunks_path=settings.paths.knowledge_base_processed,
                bm25_k=top_k,
                bm25_index_path=getattr(settings.paths, "bm25_index_path", None),
            )

        raise ValueError(f"Unsupported retriever type: {retriever_name}")