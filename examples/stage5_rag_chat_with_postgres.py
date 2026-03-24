import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.agents import RAGChatWithPostgres
from agentrf.llm import LLMFactory
from agentrf.rag_retrievers import RetrieverFactory
from agentrf.settings import load_settings

load_dotenv()

config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "5432")
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


if __name__ == "__main__":
    thread_id = input("Thread ID: ").strip() or "demo-thread-1"

    llm = LLMFactory.create(
        provider=settings.rag.llm.provider,
        model=settings.rag.llm.model,
        base_url=settings.rag.llm.base_url,
        temperature=settings.rag.llm.temperature,
    )

    embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model)

    retriever = RetrieverFactory.create(
        retriever_type=settings.rag.retriever.type,
        top_k=settings.rag.retriever.top_k,
        chroma_persist_dir=settings.paths.chroma_db_dir,
        chunks_path=settings.paths.knowledge_base_processed,
        embedding_function=embeddings,
        bm25_index_path=settings.paths.bm25_index_path,
    )

    agent = RAGChatWithPostgres(
        llm=llm,
        retriever=retriever,
        system_prompt=settings.rag.prompt.system,
        db_uri=db_uri,
        top_k=settings.rag.retriever.top_k,
    )

    print(f"RAG Chat with Postgres Memory ({settings.rag.retriever.type}, thread_id={thread_id})")

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            print("\nAssistant:\n")
            for text in agent.stream_answer(
                user_input=user_input,
                thread_id=thread_id,
            ):
                print(text, end="", flush=True)
            print("\n")
    finally:
        agent.close()
