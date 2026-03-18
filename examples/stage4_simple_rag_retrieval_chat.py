import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.rag_retrievers import VectorRetriever
from agentrf.llm import LLMFactory
from agentrf.settings import load_settings


load_dotenv()

config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model)

retriever = VectorRetriever(
    chroma_persist_dir=settings.paths.chroma_db_dir,
    embedding_function=embeddings,
    vector_k=settings.rag.retriever.top_k,
)

llm = LLMFactory.create(
    provider=settings.rag.llm.provider,
    model=settings.rag.llm.model,
    base_url=settings.rag.llm.base_url,
    temperature=settings.rag.llm.temperature,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", settings.rag.prompt.system),
    ("human", "Kontext:\n{context}\n\nFrage: {query}\n\nAntwort:")
])


def run(user_query: str, llm, retriever, prompt, top_k: int):
    docs = retriever.retrieve(user_query, top_k=top_k)

    if not docs:
        context = "Keine relevanten Dokumente gefunden."
        print("No documents retrieved.")
    else:
        context = "\n\n".join(
            f"--- {doc.metadata.get('filename', 'Unknown')} ---\n{doc.page_content}"
            for doc in docs
        )
        print(f"Retrieved {len(docs)} document(s).")

    chain = prompt | llm

    print("\nAssistant:\n")
    for chunk in chain.stream({"query": user_query, "context": context}):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("Simple RAG Retrieval Chat")

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