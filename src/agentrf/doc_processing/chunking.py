# helper functions to chunk processed markdown documents into canonical retrieval chunks

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into canonical retrieval chunks.

    Adds:
    - source
    - filename
    - chunk_id
    - total_chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunked_docs: List[Document] = []

    for doc in docs:
        source = str(doc.metadata.get("source", "")) if doc.metadata else ""
        filename = Path(source).name if source else ""

        splits = splitter.split_documents([doc])

        total_chunks = len(splits)
        for idx, chunk in enumerate(splits):
            if chunk.metadata is None:
                chunk.metadata = {}

            if source:
                chunk.metadata["source"] = source
            if filename:
                chunk.metadata["filename"] = filename

            chunk.metadata["chunk_id"] = idx
            chunk.metadata["total_chunks"] = total_chunks

            chunked_docs.append(chunk)

    return chunked_docs
