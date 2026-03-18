import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.rag_retrievers import HybridRetriever
from agentrf.llm import LLMFactory

# 1. Load Config
from agentrf.settings import load_settings
from dotenv import load_dotenv
import os

# 1. Load the config dynamically
load_dotenv()
config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

chroma_db_dir = settings.paths.chroma_db_dir
knowledge_base_processed = settings.paths.knowledge_base_processed
vector_top_k = settings.rag.retriever.top_k
bm25_top_k = settings.rag.retriever.top_k
prompt = settings.rag.prompt.system

# 2. Initialize Retriever & LLM using Configs
embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model)
retriever = HybridRetriever(
    chroma_persist_dir=chroma_db_dir,
    processed_kb_dir=knowledge_base_processed,
    embedding_function=embeddings,
    vector_k=vector_top_k,
    bm25_k=bm25_top_k
)

llm = LLMFactory.create(
    provider=settings.rag.llm.provider,
    model=settings.rag.llm.model,
    base_url=settings.rag.llm.base_url,
    temperature=settings.rag.llm.temperature,
)

# 3. Setup Prompt from Config
prompt = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("human", "Kontext:\n{context}\n\nFrage: {query}\n\nAntwort:")
])

def run(user_query: str):
    print("\n[Retriever] Searching...", flush=True)
    top_k = vector_top_k
    docs = retriever.retrieve(user_query, top_k=top_k)

    if not docs:
        context = "Keine relevanten Dokumente gefunden."
    else:
        context = "\n\n".join(
            f"--- {d.metadata.get('filename', 'Unknown')} ---\n{d.page_content}"
            for d in docs
        )
        print(f"  → {len(docs)} documents retrieved:")
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            print(f"    [{i}] {meta.get('filename', 'unknown')} | chunk: {meta.get('chunk_id', 'n/a')}")

    print("\n[LLM] Generating answer...\n", flush=True)
    
    # Modern LCEL Streaming
    chain = prompt | llm
    for chunk in chain.stream({"query": user_query, "context": context}):
        print(chunk.content, end="", flush=True)
    print("\n")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing RAG Chat System...")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        run(user_input)