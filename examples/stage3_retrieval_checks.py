from agentrf.rag_retrievers import VectorRetriever, BM25Retriever, HybridRetriever
from agentrf.settings import load_settings
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()
config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

print("Loading HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name= settings.rag.embedding.model,
    cache_folder=os.getenv("HF_HOME")
)
vector = VectorRetriever(
    chroma_persist_dir=settings.paths.chroma_db_dir,
    embedding_function=embeddings,
    vector_k=settings.rag.retriever.top_k,
)

bm25 = BM25Retriever(
    chunks_path=settings.paths.chunks_path,
    bm25_index_path=settings.paths.bm25_index_path,
    bm25_k=settings.rag.retriever.top_k,
)

hybrid = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)

docs = hybrid.retrieve("Was sind die Sicherheitsanforderungen?", top_k=5)

print(f"Retrieved {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    filename = doc.metadata.get("filename")
    chunk_id = doc.metadata.get("chunk_id")
    print(f"{i}. {filename} (chunk_id: {chunk_id})\n{doc.page_content}\n")  

docs = bm25.retrieve("Was sind die Sicherheitsanforderungen?", top_k=5)
print(f"Retrieved {len(docs)} documents with bm25:")
for i, doc in enumerate(docs, 1):
    filename = doc.metadata.get("filename")
    chunk_id = doc.metadata.get("chunk_id")
    print(f"{i}. {filename} (chunk_id: {chunk_id})\n{doc.page_content[:200]}\n")  