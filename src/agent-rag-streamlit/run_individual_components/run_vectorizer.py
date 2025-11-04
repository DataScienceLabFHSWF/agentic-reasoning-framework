# In your application file
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from data_utils.chroma_db_from_md import create_chromadb_from_markdown, load_hf_embeddings_from_env

# 1. Load the embedding model configuration from your .env file
embedding_model = load_hf_embeddings_from_env()

# 2. Call the main function with the configured model
my_vector_store = create_chromadb_from_markdown(
    folder_path="../processed_files",
    embedding_model=embedding_model,
    persist_directory="../chroma_db"
)

if my_vector_store:
    print("Vector store is ready for use.")