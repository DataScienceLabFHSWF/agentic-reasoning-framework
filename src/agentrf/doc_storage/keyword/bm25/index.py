# utilities for building, saving, and loading a BM25 retriever over canonical chunks

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Build, save, and load a BM25 retriever over canonical chunks.
    """

    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever

    @classmethod
    def build(cls, chunks: List[Document], k: int = 5) -> "BM25Index":
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = k
        return cls(retriever=retriever)

    def save(self, path: str | Path) -> None:
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as f:
            pickle.dump(self.retriever, f)

        logger.info("Saved BM25 index to %s", output_path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        input_path = Path(path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {input_path}")

        with input_path.open("rb") as f:
            retriever = pickle.load(f)

        if not isinstance(retriever, BM25Retriever):
            raise TypeError("Loaded object is not a BM25Retriever")

        logger.info("Loaded BM25 index from %s", input_path)
        return cls(retriever=retriever)