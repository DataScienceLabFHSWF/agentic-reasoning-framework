import os
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.llm import LLMFactory
from agentrf.rag_retrievers import VectorRetriever
from agentrf.settings import load_settings


load_dotenv()

config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

llm = LLMFactory.create(
    provider=settings.rag.llm.provider,
    model=settings.rag.llm.model,
    base_url=settings.rag.llm.base_url,
    temperature=settings.rag.llm.temperature,
)

embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model)

retriever = VectorRetriever(
    chroma_persist_dir=settings.paths.chroma_db_dir,
    embedding_function=embeddings,
    vector_k=settings.rag.retriever.top_k,
)


if __name__ == "__main__":
    print("Initializing simple RAG chat...")

    print("[System] Measuring first LLM response...", flush=True)
    warmup_start = time.perf_counter()
    warmup_response = llm.invoke("Hi")
    warmup_end = time.perf_counter()

    warmup_text = getattr(warmup_response, "content", str(warmup_response))
    print(f"[System] First response: {warmup_text}")
    print(f"[Timing] First LLM call took: {warmup_end - warmup_start:.3f} s\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        retrieval_start = time.perf_counter()
        docs = retriever.retrieve(user_input, top_k=settings.rag.retriever.top_k)
        retrieval_end = time.perf_counter()
        print(f"[Timing] Retrieval took: {retrieval_end - retrieval_start:.3f} s")
        

        context_start = time.perf_counter()
        context = "\n\n".join([d.page_content for d in docs])
        print("##################################\n")
        print("Conext:")
        print(context)
        print("##################################\n")
        rag_input = f"Context:\n{context}\n\nQuestion:\n{user_input}"
        context_end = time.perf_counter()
        print(f"[Timing] Context building took: {context_end - context_start:.3f} s")

        start = time.perf_counter()
        print("Assistant: ", end="", flush=True)

        for chunk in llm.stream(rag_input):
            text = getattr(chunk, "content", str(chunk))
            if text:
                print(text, end="", flush=True)

        end = time.perf_counter()
        print()
        print(f"[Timing] Response took: {end - start:.3f} s\n")