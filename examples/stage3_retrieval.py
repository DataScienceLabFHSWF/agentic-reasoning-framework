import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Import the pure framework code
from agentrf.rag_retrievers import HybridRetriever
from agentrf.settings import load_settings
from dotenv import load_dotenv
import os

# 1. Load the config dynamically
load_dotenv()
config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

processed_kb_dir = settings.paths.knowledge_base_processed
chroma_db_dir = settings.paths.chroma_db_dir

# 2. Initialize Embeddings
print("Loading HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name= settings.rag.embedding.model,
    cache_folder=os.getenv("HF_HOME")
)

# 3. Initialize AgentRF's Hybrid Retriever
print("Initializing Hybrid Retriever...")
retriever = HybridRetriever(
    chroma_persist_dir=chroma_db_dir,
    processed_kb_dir=processed_kb_dir,
    embedding_function=embeddings,
    vector_k=3,
    bm25_k=2
)

# 4. Perform a hybrid search
query = "What are the safety protocols for the facility?"
print(f"\nSearching for: '{query}'")

results = retriever.retrieve(query, top_k=4)

for i, doc in enumerate(results, 1):
    source = doc.metadata.get('filename', 'Unknown')
    chunk_id = doc.metadata.get('chunk_id', '?')
    print(f"\n--- Result {i} (Source: {source}, Chunk: {chunk_id}) ---")
    print(doc.page_content[:200] + "...")