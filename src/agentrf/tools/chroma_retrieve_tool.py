from functools import lru_cache

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.rag_retrievers import VectorRetriever


@lru_cache(maxsize=4)
def get_embedding_function(model_name: str, cache_folder: str | None = None):
    """
    Load and cache the embedding model once per unique argument combination.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_folder,
    )


def build_chroma_retriever_tool(
    chroma_persist_dir: str,
    model_name: str,
    top_k: int = 5,
    cache_folder: str | None = None,
):
    """
    Build an LLM-facing retrieval tool backed by the project's VectorRetriever.
    """
    embedding_function = get_embedding_function(
        model_name=model_name,
        cache_folder=cache_folder,
    )

    vector_retriever = VectorRetriever(
        chroma_persist_dir=chroma_persist_dir,
        embedding_function=embedding_function,
        vector_k=top_k,
    )

    @tool("retrieve_information")
    def retrieve_information(query: str) -> str:
        """
        Retrieve relevant information from the indexed knowledge base.
        Use this tool when you need grounded facts, evidence, definitions,
        or document-based context.
        """
        docs = vector_retriever.retrieve(query)

        if not docs:
            return "No relevant documents found."

        return "\n\n".join(
            f"--- {doc.metadata.get('filename', 'Unknown')} "
            f"(chunk_id: {doc.metadata.get('chunk_id', '?')}) ---\n"
            f"{doc.page_content}"
            for doc in docs
        )

    return retrieve_information