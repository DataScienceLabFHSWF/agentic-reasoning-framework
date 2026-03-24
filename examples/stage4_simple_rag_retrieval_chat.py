import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.rag_retrievers import VectorRetriever, BM25Retriever, HybridRetriever
from agentrf.llm import LLMFactory
from agentrf.settings import load_settings


load_dotenv()

config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model)


def build_retriever(settings, embeddings):
    retriever_type = settings.rag.retriever.type.lower()
    top_k = settings.rag.retriever.top_k

    if retriever_type == "vector":
        return VectorRetriever(
            chroma_persist_dir=settings.paths.chroma_db_dir,
            embedding_function=embeddings,
            vector_k=top_k,
        )

    if retriever_type == "bm25":
        return BM25Retriever(
            chunks_path=settings.paths.chunks_path,
            bm25_index_path=settings.paths.bm25_index_path,
            bm25_k=top_k,
        )

    if retriever_type == "hybrid":
        vector_retriever = VectorRetriever(
            chroma_persist_dir=settings.paths.chroma_db_dir,
            embedding_function=embeddings,
            vector_k=top_k,
        )
        bm25_retriever = BM25Retriever(
            chunks_path=settings.paths.chunks_path,
            bm25_index_path=settings.paths.bm25_index_path,
            bm25_k=top_k,
        )
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
        )

    raise ValueError(f"Unsupported retriever type: {retriever_type}")


retriever = build_retriever(settings, embeddings)

llm = LLMFactory.create(
    provider=settings.rag.llm.provider,
    model=settings.rag.llm.model,
    base_url=settings.rag.llm.base_url,
    temperature=settings.rag.llm.temperature,
)

prompt = ChatPromptTemplate.from_messages(
    [("system", settings.rag.prompt.system), ("human", "Kontext:\n{context}\n\nFrage: {query}\n\nAntwort:")]
)


def run(user_query: str, llm, retriever, prompt, top_k: int):
    docs = retriever.retrieve(user_query, top_k=top_k)

    if not docs:
        context = "Keine relevanten Dokumente gefunden."
        print("No documents retrieved.")
    else:
        context = "\n\n".join(
            f"--- {doc.metadata.get('filename', 'Unknown')} (chunk {doc.metadata.get('chunk_id', '?')}) ---\n{doc.page_content}"
            for doc in docs
        )
        print(f"Retrieved {len(docs)} document(s).")

    chain = prompt | llm

    print("\nAssistant:\n")
    for chunk in chain.stream({"query": user_query, "context": context}):
        text = getattr(chunk, "content", str(chunk))
        if text:
            print(text, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print(f"Simple RAG Retrieval Chat ({settings.rag.retriever.type})")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input:
            run(
                user_query=user_input,
                llm=llm,
                retriever=retriever,
                prompt=prompt,
                top_k=settings.rag.retriever.top_k,
            )
