import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_utils.retriever import hybrid_retrieve 
from langchain.schema import Document

# --- Define input ---
query = "Welche Regelungen gelten für die Lagerung von Atommüll in Deutschland??"
chroma_dir = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"
processed_dir = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"

# --- Call hybrid retrieval ---
retrieved_docs: list[Document] = hybrid_retrieve(query, chroma_dir, processed_dir, k=2)

# --- Inspect results ---
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:300])  # Print first 300 characters
