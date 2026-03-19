from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import yaml


@dataclass
class PathSettings:
    knowledge_base_raw: Path
    knowledge_base_processed: Path
    uploads_tmp: Path
    chroma_db_dir: Path
    chunks_path: Path
    bm25_index_path: Path


@dataclass
class ProcessingSettings:
    save_markdown: bool
    supported_extensions: List[str]


@dataclass
class RetrieverSettings:
    type: str
    top_k: int


@dataclass
class ChunkingSettings:
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class EmbeddingSettings:
    model: str


@dataclass
class LLMSettings:
    provider: str
    model: str
    base_url: str
    temperature: float = 0.0


@dataclass
class PromptSettings:
    system: str


@dataclass
class RAGSettings:
    retriever: RetrieverSettings
    chunking: ChunkingSettings
    embedding: EmbeddingSettings
    llm: LLMSettings
    prompt: PromptSettings


@dataclass
class Settings:
    paths: PathSettings
    processing: ProcessingSettings
    rag: RAGSettings

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Settings":
        return cls(
            paths=PathSettings(
                knowledge_base_raw=Path(raw["paths"]["knowledge_base_raw"]),
                knowledge_base_processed=Path(raw["paths"]["knowledge_base_processed"]),
                uploads_tmp=Path(raw["paths"]["uploads_tmp"]),
                chroma_db_dir=Path(raw["paths"]["chroma_db_dir"]),
                chunks_path=Path(raw["paths"]["chunks_path"]),
                bm25_index_path=Path(raw["paths"]["bm25_index_path"]),
            ),
            processing=ProcessingSettings(**raw["processing"]),
            rag=RAGSettings(
                retriever=RetrieverSettings(**raw["rag"]["retriever"]),
                chunking=ChunkingSettings(**raw["rag"]["chunking"]),
                embedding=EmbeddingSettings(**raw["rag"]["embedding"]),
                llm=LLMSettings(**raw["rag"]["llm"]),
                prompt=PromptSettings(**raw["rag"]["prompt"]),
            ),
        )


def load_settings(path: str | Path) -> Settings:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.from_dict(raw)