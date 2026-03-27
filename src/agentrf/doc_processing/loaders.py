# helper functions to load processed markdown documents into LangChain Document format

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document


def load_processed_markdown(processed_dir: str | Path) -> List[Document]:
    """
    Load processed markdown files from a directory.

    This assumes DocProcessor has already converted raw documents into .md files.
    """
    processed_path = Path(processed_dir).resolve()

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed markdown directory not found: {processed_path}")
    if not processed_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {processed_path}")

    loader = DirectoryLoader(
        str(processed_path),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    )
    return loader.load()
