# IO utilities for persisting and loading canonical chunks as JSONL.

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def save_chunks_jsonl(chunks: List[Document], path: str | Path) -> None:
    """
    Persist canonical chunks to disk as JSONL.
    """
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata or {},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str | Path) -> List[Document]:
    """
    Load canonical chunks from JSONL.
    """
    input_path = Path(path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")

    chunks: List[Document] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            chunks.append(
                Document(
                    page_content=raw["page_content"],
                    metadata=raw.get("metadata", {}),
                )
            )

    return chunks
