import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Optional
from dotenv import load_dotenv

def load_hf_embeddings_from_env() -> HuggingFaceEmbeddings:
    """
    Loads environment variables from a .env file and creates a
    HuggingFaceEmbeddings instance.

    Returns:
        HuggingFaceEmbeddings: A configured embedding model.
    """
    load_dotenv()  # Load environment variables from .env file

    # Get the cache folder path from the environment variables
    # We use os.getenv to safely retrieve the value
    cache_folder = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if not cache_folder:
        raise ValueError("Hugging Face cache folder environment variable not set. Please set HF_HOME, TRANSFORMERS_CACHE, or HUGGINGFACE_HUB_CACHE in your .env file.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        cache_folder=cache_folder,
    )
    return embedding_model


def create_chromadb_from_markdown(
    folder_path: str,
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str = "chroma_db",
) -> Optional[Chroma]:
    """
    Loads markdown files from a folder, splits them into chunks, and
    creates a ChromaDB vector store using a pre-configured embedding model.

    Args:
        folder_path (str): The path to the folder containing markdown files.
        embedding_model (HuggingFaceEmbeddings): A pre-configured HuggingFaceEmbeddings object.
        persist_directory (str): The directory to save the ChromaDB vector store.

    Returns:
        Optional[Chroma]: The created ChromaDB vector store instance, or None if an error occurs.
    """
    # Check if the persistence directory exists and create it if not
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        print(f"Created directory: {persist_directory}")

    markdown_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".md")]
    if not markdown_files:
        print("No markdown files found in the specified folder.")
        return None

    # Load the documents
    docs = []
    for file_path in markdown_files:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    # Create the ChromaDB vector store with the provided embedding model
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vector_store.persist()
        print(f"Successfully created and persisted ChromaDB vector store at '{persist_directory}' with {len(chunks)} chunks.")
        return vector_store
    except Exception as e:
        print(f"Error creating ChromaDB: {e}")
        return None