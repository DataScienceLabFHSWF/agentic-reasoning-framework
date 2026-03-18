import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import the pure framework code
from agentrf.doc_storage import ChromaManager

from agentrf.settings import load_settings
from dotenv import load_dotenv
import os

# 1. Load the config dynamically
load_dotenv()
config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

processed_kb_dir = settings.paths.knowledge_base_processed
chroma_db_dir = settings.paths.chroma_db_dir

# 2. Initialize Embeddings (Consumer controls the model)
print("Loading HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name= settings.rag.embedding.model,
    cache_folder=os.getenv("HF_HOME")
)

# 3. Initialize AgentRF's ChromaManager
vector_store = ChromaManager(
    persist_directory=chroma_db_dir,
    embedding_function=embeddings
)

# 4a. Add a full directory of processed Markdown files
print(f"Ingesting processed KB from: {processed_kb_dir}")
added_ids = vector_store.add_directory(processed_kb_dir)
print(f"Successfully added {len(added_ids)} chunks to the database.")

# 4b. Or add a single file later on
single_file = processed_kb_dir / "uvu.md"
print(f"Adding single file: {single_file}")
single_file_ids = vector_store.add_file(single_file)
print(f"Successfully added {len(single_file_ids)} chunks from {single_file.name} to the database.")