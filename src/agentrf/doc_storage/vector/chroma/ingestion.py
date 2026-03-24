# Manages ingestion of already-chunked documents into a persistent Chroma DB.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


class ChromaManager:
    """
    Manages ingestion of already-chunked documents into a persistent Chroma DB.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_function: Embeddings,
    ) -> None:
        self.persist_directory = str(Path(persist_directory).resolve())
        self.embedding_function = embedding_function

        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )

    def add_chunks(self, chunks: List[Document]) -> List[str]:
        """
        Add a list of canonical chunks to Chroma.
        """
        if not chunks:
            logger.warning("No chunks provided to ChromaManager.add_chunks()")
            return []

        ids = self.db.add_documents(chunks)
        logger.info("Added %d chunks to Chroma DB", len(ids))
        return ids
