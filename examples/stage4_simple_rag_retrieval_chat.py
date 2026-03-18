import os
import time

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.rag_retrievers import VectorRetriever
from agentrf.llm import LLMFactory
from agentrf.settings import load_settings


load_dotenv()

config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

chroma_db_dir = settings.paths.chroma_db_dir
knowledge_base_processed = settings.paths.knowledge_base_processed
vector_top_k = settings.rag.retriever.top_k
bm25_top_k = settings.rag.retriever.top_k
system_prompt = settings.rag.prompt.system

embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model)

retriever = VectorRetriever(
    chroma_persist_dir=chroma_db_dir,
    embedding_function=embeddings,
    vector_k=vector_top_k,
)


llm = LLMFactory.create(
    provider=settings.rag.llm.provider,
    model=settings.rag.llm.model,
    base_url=settings.rag.llm.base_url,
    temperature=settings.rag.llm.temperature,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Kontext:\n{context}\n\nFrage: {query}\n\nAntwort:")
])


def run(user_query: str, llm, retriever, prompt, top_k: int):
    print("\n[Retriever] Searching...", flush=True)

    retrieval_start = time.perf_counter()
    docs = retriever.retrieve(user_query, top_k=top_k)
    retrieval_end = time.perf_counter()

    if not docs:
        context = "Keine relevanten Dokumente gefunden."
        print("  → 0 documents retrieved.")
    else:
        context = "\n\n".join(
            f"--- {d.metadata.get('filename', 'Unknown')} ---\n{d.page_content}"
            for d in docs
        )
        print(f"  → {len(docs)} documents retrieved:")
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            print(
                f"    [{i}] {meta.get('filename', 'unknown')} | "
                f"chunk: {meta.get('chunk_id', 'n/a')}"
            )

    print(f"[Timing] Retrieval time: {retrieval_end - retrieval_start:.3f} s")
    print("\n[LLM] Generating answer...\n")

    chain = prompt | llm

    generation_start = time.perf_counter()
    first_token_time = None

    for chunk in chain.stream({"query": user_query, "context": context}):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        if chunk.content:
            print(chunk.content, end="", flush=True)

    generation_end = time.perf_counter()
    print("\n")

    if first_token_time is not None:
        print(f"[Timing] Time to first token: {first_token_time - generation_start:.3f} s")
    print(f"[Timing] Total generation time: {generation_end - generation_start:.3f} s")


if __name__ == "__main__":
    print("Initializing RAG Chat System...")

    print("[System] Measuring first LLM response...", flush=True)
    warmup_start = time.perf_counter()
    warmup_response = llm.invoke("Hi")
    warmup_end = time.perf_counter()

    warmup_text = getattr(warmup_response, "content", str(warmup_response))
    print(f"[System] First response: {warmup_text}")
    print(f"[Timing] First LLM call took: {warmup_end - warmup_start:.3f} s\n")

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
                top_k=vector_top_k,
            )