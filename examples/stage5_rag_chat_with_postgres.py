import os

from dotenv import load_dotenv

from agentrf.agents import RAGChatWithPostgres
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

    agent = RAGChatWithPostgres(settings=settings, db_uri=db_uri)

    print(
        f"RAG Chat with Postgres Memory "
        f"({settings.rag.retriever.type}, thread_id={thread_id})"
    )

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