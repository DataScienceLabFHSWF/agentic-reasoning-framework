import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Optional, List
from langchain_core.documents import Document
from dotenv import load_dotenv

def load_hf_embeddings_from_env() -> HuggingFaceEmbeddings:
    """
    Loads environment variables from a .env file and creates a
    HuggingFaceEmbeddings instance.

    Returns:
        HuggingFaceEmbeddings: A configured embedding model.
    """
    # Find .env file - look in src directory (two levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(current_dir))
    env_path = os.path.join(src_dir, ".env")
    
    if not os.path.exists(env_path):
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from: {env_path}")

    # Get the cache folder path from the environment variables
    cache_folder = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("TRANSFORMERS_CACHE")
    if not cache_folder:
        raise ValueError("Hugging Face cache folder environment variable not set. Please set HF_HOME, TRANSFORMERS_CACHE, or HUGGINGFACE_HUB_CACHE in your .env file.")

    # Convert relative path to absolute path if needed
    if not os.path.isabs(cache_folder):
        cache_folder = os.path.abspath(os.path.join(src_dir, cache_folder))
    
    # Ensure the cache directory exists
    os.makedirs(cache_folder, exist_ok=True)
    print(f"Using HuggingFace cache directory: {cache_folder}")

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
    Loads markdown files from a folder, splits them into chunks with stable chunk IDs,
    and creates a ChromaDB vector store using a pre-configured embedding model.

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

    # Load documents and assign chunk IDs
    all_chunks = []
    
    for file_path in markdown_files:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            for doc in docs:
                chunks = text_splitter.split_documents([doc])
                
                # Assign sequential chunk IDs per source file
                for i, chunk in enumerate(chunks):
                    # Ensure metadata exists
                    if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                        chunk.metadata = {}
                    
                    # Set stable identifiers
                    chunk.metadata['source'] = file_path
                    chunk.metadata['filename'] = os.path.basename(file_path)
                    chunk.metadata['chunk_id'] = i  # Sequential within each file
                    chunk.metadata['total_chunks'] = len(chunks)  # Total chunks in this file
                    
                    all_chunks.append(chunk)
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not all_chunks:
        print("No chunks were created from the markdown files.")
        return None

    # Create the ChromaDB vector store with chunk-level documents
    try:
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vector_store.persist()
        print(f"Successfully created and persisted ChromaDB vector store at '{persist_directory}' with {len(all_chunks)} chunks from {len(markdown_files)} files.")
        return vector_store
    except Exception as e:
        print(f"Error creating ChromaDB: {e}")
        return None

def get_chunk_neighbors(
    source_file: str, 
    chunk_id: int, 
    processed_dir: str, 
    window_size: int = 1
) -> List[Document]:
    """
    Retrieve neighboring chunks around a specific chunk for context expansion.
    
    Args:
        source_file (str): Path to the source file
        chunk_id (int): ID of the target chunk
        processed_dir (str): Directory containing processed files
        window_size (int): Number of chunks before and after to include
        
    Returns:
        List[Document]: List of neighboring chunks with metadata
    """
    try:
        # Load the specific source file
        loader = UnstructuredMarkdownLoader(source_file)
        docs = loader.load()
        
        if not docs:
            return []
            
        # Split into chunks the same way as during ingestion
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(docs)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['source'] = source_file
            chunk.metadata['filename'] = os.path.basename(source_file)
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        # Extract window around target chunk
        start_idx = max(0, chunk_id - window_size)
        end_idx = min(len(chunks), chunk_id + window_size + 1)
        
        return chunks[start_idx:end_idx]
        
    except Exception as e:
        print(f"Error retrieving neighbors for {source_file}, chunk {chunk_id}: {e}")
        return []