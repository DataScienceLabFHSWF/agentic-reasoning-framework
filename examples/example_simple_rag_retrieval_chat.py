import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from agentrf.rag_retrievers import HybridRetriever
from agentrf.llm import LLMFactory

# 1. Load Config
with open("/home/rrao/projects/agents/agentic-reasoning-framework/src/config/toy.yaml", "r") as f:
    config = yaml.safe_load(f)

paths_cfg = config["paths"]
rag_cfg = config["rag"]

# 2. Initialize Retriever & LLM using Configs
embeddings = HuggingFaceEmbeddings(model_name=rag_cfg["embedding"]["model"])
retriever = HybridRetriever(
    chroma_persist_dir=paths_cfg["chroma_db_dir"],
    processed_kb_dir=paths_cfg["knowledge_base_processed"],
    embedding_function=embeddings,
    vector_k=rag_cfg["retriever"]["top_k"],
    bm25_k=rag_cfg["retriever"]["top_k"]
)

llm = LLMFactory.create(rag_cfg["llm"])

# 3. Setup Prompt from Config
prompt = ChatPromptTemplate.from_messages([
    ("system", rag_cfg["prompt"]["system"]),
    ("human", "Kontext:\n{context}\n\nFrage: {query}\n\nAntwort:")
])

def run(user_query: str):
    print("\n[Retriever] Searching...", flush=True)
    top_k = rag_cfg["retriever"]["top_k"]
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