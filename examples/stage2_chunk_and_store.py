import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

from agentrf.doc_processing import chunk_documents, load_processed_markdown
from agentrf.doc_storage import (
    BM25Index,
    ChromaManager,
    save_chunks_jsonl,
)
from agentrf.settings import load_settings

load_dotenv()
config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

print("Loading HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model, cache_folder=os.getenv("HF_HOME"))

docs = load_processed_markdown(settings.paths.knowledge_base_processed)
chunks = chunk_documents(
    docs,
    chunk_size=settings.rag.chunking.chunk_size,
    chunk_overlap=settings.rag.chunking.chunk_overlap,
)

save_chunks_jsonl(chunks, settings.paths.chunks_path)

chroma = ChromaManager(
    persist_directory=settings.paths.chroma_db_dir,
    embedding_function=embeddings,
)
chroma.add_chunks(chunks)

bm25_index = BM25Index.build(chunks, k=settings.rag.retriever.top_k)
bm25_index.save(settings.paths.bm25_index_path)
