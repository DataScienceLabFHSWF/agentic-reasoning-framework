from langchain_huggingface import HuggingFaceEmbeddings
from agentrf.pipelines import DocumentIngestionPipeline
from agentrf.settings import load_settings
from dotenv import load_dotenv
import os

load_dotenv()

config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

embeddings = HuggingFaceEmbeddings(model_name=settings.rag.embedding.model)

pipeline = DocumentIngestionPipeline(
    processed_dir=str(settings.paths.knowledge_base_processed),
    chunks_path=str(settings.paths.chunks_path),
    chroma_persist_dir=str(settings.paths.chroma_db_dir),
    embedding_function=embeddings,
    bm25_index_path=str(settings.paths.bm25_index_path),
    chunk_size=settings.rag.chunking.chunk_size,
    chunk_overlap=settings.rag.chunking.chunk_overlap,
)

result = pipeline.ingest_document(str(settings.paths.knowledge_base_raw / "filename.pdf"))
print(result)
