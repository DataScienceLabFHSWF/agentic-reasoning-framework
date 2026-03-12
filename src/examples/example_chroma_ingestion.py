import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import the pure framework code
from agentrf.doc_storage.vector.chroma.ingestion import ChromaManager

# 1. Load application environment and config
load_dotenv("/home/rrao/projects/agents/agentic-reasoning-framework/src/.env") # Only the consumer app cares about .env
config_file = "/home/rrao/projects/agents/agentic-reasoning-framework/src/config/toy.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

processed_kb_dir = Path(config["paths"]["knowledge_base_processed"])
chroma_db_dir = Path(config["paths"].get("chroma_db_dir", "/tmp/chroma_db"))

# 2. Initialize Embeddings (Consumer controls the model)
print("Loading HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", # Or read from env
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