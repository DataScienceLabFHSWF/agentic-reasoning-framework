from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class LLMSettings:
    provider: str
    model: str
    base_url: str
    temperature: float = 0.0


@dataclass
class EmbeddingSettings:
    model_name: str


@dataclass
class RetrieverSettings:
    vector_k: int = 4
    bm25_k: int = 4
    final_k: int = 4


@dataclass
class PathSettings:
    chroma_dir: Path
    processed_dir: Path
    raw_dir: Path | None = None


@dataclass
class Settings:
    llm: LLMSettings
    embeddings: EmbeddingSettings
    retriever: RetrieverSettings
    paths: PathSettings

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Settings":
        return cls(
            llm=LLMSettings(**raw["llm"]),
            embeddings=EmbeddingSettings(**raw["embeddings"]),
            retriever=RetrieverSettings(**raw.get("retriever", {})),
            paths=PathSettings(
                chroma_dir=Path(raw["paths"]["chroma_dir"]),
                processed_dir=Path(raw["paths"]["processed_dir"]),
                raw_dir=Path(raw["paths"]["raw_dir"]) if raw["paths"].get("raw_dir") else None,
            ),
        )


def load_settings(path: str | Path) -> Settings:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.from_dict(raw)