import sys
import os
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables from a .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_utils.retriever import (
    hybrid_retrieve, 
    RetrievalConfig, 
    retrieve_with_expansion,
    retrieve_with_mmr,
    retrieve_hierarchical,
    check_chromadb_chunk_compatibility
)

# --- Define input ---
query = "Welche Regelungen gelten für die Lagerung von Atommüll in Deutschland??"
query = "Wer sind die Genehmigungsinhaber (Antragsteller) der Anlage KRB II?"
query = "Wie viele Blöcke gehören zur Anlage KRB II und welche elektrische Leistung hat jeder Block?"
chroma_dir = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"
processed_dir = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"

# --- Check ChromaDB compatibility ---
print("=== ChromaDB Compatibility Check ===")
is_chunk_aware = check_chromadb_chunk_compatibility(chroma_dir)
if is_chunk_aware:
    print("✓ ChromaDB contains chunk-aware documents")
else:
    print("⚠ ChromaDB contains legacy documents ohne chunk IDs")
    print("  Erwägen Sie, ChromaDB mit dem aktualisierten Ingestion-Skript neu zu generieren, um die volle Funktionalität zu gewährleisten")

# --- Test different retrieval strategies ---

print("\n=== Standard Hybrid Retrieval ===")
retrieved_docs: list[Document] = hybrid_retrieve(query, chroma_dir, processed_dir, k=5, vector_weight=0.7, bm25_weight=0.3)

print(f"Anzahl der abgerufenen Dokumente: {len(retrieved_docs)}")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n--- Dokument {i} ---")

    score        = float(doc.metadata.get('score', 0.0))
    vector_score = float(doc.metadata.get('vector_score', 0.0))
    bm25_score   = float(doc.metadata.get('bm25_score', 0.0))

    src      = doc.metadata.get('source', '')
    filename = doc.metadata.get('filename') or (os.path.basename(src) if src else '')
    chunk_id = doc.metadata.get('chunk_id', 'unknown')
    total_chunks = doc.metadata.get('total_chunks', 'unknown')
    is_legacy = doc.metadata.get('chunk_id_inferred', False)

    legacy_marker = " [LEGACY]" if is_legacy else ""
    
    print(f"Datei: {filename or '[unbekannt]'}{legacy_marker}")
    if src:
        print(f"Pfad: {src}")
    print(f"Chunk: {chunk_id}/{total_chunks}")
    print(f"Gesamtpunktzahl: {score:.3f} (Vektor: {vector_score:.3f}, BM25: {bm25_score:.3f})")
    print(doc.page_content[:300])

print("\n" + "="*50)
print("=== Retrieval with Context Expansion ===")
if not is_chunk_aware:
    print("⚠ Die Kontextvergrößerung ist möglicherweise eingeschränkt, wenn veraltete Dokumente verwendet werden")

expanded_docs = retrieve_with_expansion(query, chroma_dir, processed_dir, k=3, window_size=1)

for i, doc in enumerate(expanded_docs, 1):
    is_context = doc.metadata.get('is_expanded_context', False)
    chunk_id = doc.metadata.get('chunk_id', 'unknown')
    target_chunk = doc.metadata.get('target_chunk_id', chunk_id)
    is_legacy = doc.metadata.get('chunk_id_inferred', False)
    
    context_marker = " [CONTEXT]" if is_context else " [ORIGINAL]"
    legacy_marker = " [LEGACY]" if is_legacy else ""
    
    print(f"\n--- Erweitertes Dokument {i}{context_marker}{legacy_marker} ---")
    print(f"Chunk: {chunk_id} (Ziel-Chunk: {target_chunk})")
    print(f"Datei: {doc.metadata.get('filename', 'unknown')}")
    print(doc.page_content[:200])

print("\n" + "="*50)
print("=== Retrieval with MMR Diversification ===")
mmr_docs = retrieve_with_mmr(query, chroma_dir, processed_dir, k=3, diversity_threshold=0.6)

for i, doc in enumerate(mmr_docs, 1):
    print(f"\n--- MMR-Dokument {i} ---")
    print(f"Datei: {doc.metadata.get('filename', 'unknown')}")
    print(f"Chunk: {doc.metadata.get('chunk_id', 'unknown')}")
    print(doc.page_content[:200])

print("\n" + "="*50)
print("=== Hierarchical Retrieval ===")
hierarchical_docs = retrieve_hierarchical(query, chroma_dir, processed_dir, k=2)

chunk_count = 0
section_count = 0
for i, doc in enumerate(hierarchical_docs, 1):
    hierarchy_level = doc.metadata.get('hierarchy_level', 'unknown')
    is_merged = doc.metadata.get('is_merged', False)
    
    if hierarchy_level == 'chunk':
        chunk_count += 1
        doc_type = f"CHUNK {chunk_count}"
    elif hierarchy_level == 'section':
        section_count += 1
        doc_type = f"SECTION {section_count}"
    else:
        doc_type = hierarchy_level.upper()
    
    print(f"\n--- Hierarchisches Dokument {i} [{doc_type}] ---")
    if is_merged:
        start_chunk = doc.metadata.get('chunk_id_start', 'unknown')
        end_chunk = doc.metadata.get('chunk_id_end', 'unknown')
        merged_count = doc.metadata.get('merged_chunk_count', 'unknown')
        print(f"Zusammengeführte Chunks: {start_chunk}-{end_chunk} ({merged_count} Chunks)")
    else:
        print(f"Chunk: {doc.metadata.get('chunk_id', 'unknown')}")
    
    print(f"Datei: {doc.metadata.get('filename', 'unknown')}")
    print(doc.page_content[:300])
