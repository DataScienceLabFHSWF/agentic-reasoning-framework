import logging
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Manages ingestion of processed Markdown documents into a Chroma Vector Database.
    """
    def __init__(
        self, 
        persist_directory: str | Path, 
        embedding_function: Embeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_directory = str(Path(persist_directory).resolve())
        self.embedding_function = embedding_function
        
        # Initialize Chroma DB
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )
        
        # Configure the text splitter based on your original logic
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def add_file(self, file_path: str | Path) -> List[str]:
        """Loads a single processed markdown file, chunks it, and adds it to Chroma DB."""
        file_path = Path(file_path).resolve()
        
        if not file_path.is_file():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        logger.info("Processing single file for Chroma DB: %s", file_path)
        
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            docs = loader.load()
            
            if not docs:
                logger.warning("No content extracted from %s", file_path)
                return []
                
            chunks = self.text_splitter.split_documents(docs)
            
            # Enrich metadata
            for i, chunk in enumerate(chunks):
                if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata['source'] = str(file_path)
                chunk.metadata['filename'] = file_path.name
                chunk.metadata['chunk_id'] = i
                chunk.metadata['total_chunks'] = len(chunks)
            
            # Add to Chroma
            ids = self.db.add_documents(chunks)
            logger.info("Added %d chunks from %s", len(ids), file_path.name)
            return ids
            
        except Exception as e:
            logger.exception("Failed to process file %s: %s", file_path, e)
            return []

    def add_directory(self, dir_path: str | Path) -> List[str]:
        """Iterates through a directory of Markdown files and adds them to Chroma DB."""
        dir_path = Path(dir_path).resolve()
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        logger.info("Processing directory for Chroma DB: %s", dir_path)
        all_ids = []
        
        # Only process .md files (the output of DocProcessor)
        for file_path in dir_path.rglob("*.md"):
            if file_path.is_file():
                ids = self.add_file(file_path)
                all_ids.extend(ids)
                
        logger.info("Finished processing directory. Total chunks added: %d", len(all_ids))
        return all_ids