import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_utils.retriever import hybrid_retrieve 
from langchain.schema import Document

# --- Define input ---
query = "Welche Regelungen gelten für die Lagerung von Atommüll in Deutschland??"
chroma_dir = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"
processed_dir = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"

# --- Call hybrid retrieval with large k to get all documents ---
retrieved_docs: list[Document] = hybrid_retrieve(query, chroma_dir, processed_dir, k=3, vector_weight=1, bm25_weight=0)

# --- Inspect results ---
print(f"Total documents retrieved: {len(retrieved_docs)}")

for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i+1} ---")
    
    # Print scores from metadata
    score = doc.metadata.get('score', 0.0)
    vector_score = doc.metadata.get('vector_score', 0.0)
    bm25_score = doc.metadata.get('bm25_score', 0.0)
    print(f"Combined Score: {score:.3f} (Vector: {vector_score:.3f}, BM25: {bm25_score:.3f})")
    
    print(doc.page_content[:300])  # Print first 300 characters