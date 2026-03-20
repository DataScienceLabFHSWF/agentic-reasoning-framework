# utility pipeline to ingest a single raw document into the framework's retrieval stores

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agentrf.doc_processing.chunking import chunk_documents
from agentrf.doc_processing.processor import DocProcessor
from agentrf.doc_storage.chunks.io import load_chunks_jsonl, save_chunks_jsonl
from agentrf.doc_storage.keyword import BM25Index
from agentrf.doc_storage.vector.chroma.ingestion import ChromaManager

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Ingest a single raw document into the framework's retrieval stores.

    Flow:
    1. Process a single raw file into normalized markdown text
    2. Convert to one LangChain Document
    3. Chunk that document into canonical retrieval chunks
    4. Append chunks to Chroma
    5. Append chunks to canonical chunk storage (JSONL)
    6. Rebuild and save the BM25 artifact from canonical chunks
    """

    def __init__(
        self,
        processed_dir: str | Path,
        chunks_path: str | Path,
        chroma_persist_dir: str | Path,
        embedding_function: Embeddings,
        bm25_index_path: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.processed_dir = Path(processed_dir).resolve()
        self.chunks_path = Path(chunks_path).resolve()
        self.chroma_persist_dir = Path(chroma_persist_dir).resolve()
        self.bm25_index_path = Path(bm25_index_path).resolve()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.processor = DocProcessor(output_dir=self.processed_dir)
        self.chroma = ChromaManager(
            persist_directory=self.chroma_persist_dir,
            embedding_function=embedding_function,
        )

    def ingest_document(self, file_path: str | Path) -> dict[str, Any]:
        """
        Process, chunk, and index a single document into:
        - persistent Chroma vector store
        - canonical chunks JSONL
        - saved BM25 artifact

        Returns a small summary dict for logging / higher-level orchestration.
        """
        source_path = Path(file_path).resolve()

        if not source_path.is_file():
            raise FileNotFoundError(f"File not found: {source_path}")

        logger.info("Starting document ingestion for %s", source_path)

        processed = self.processor.process_document(
            file_path=source_path,
            output_dir=self.processed_dir,
            save_output=True,
        )
        if processed is None:
            raise ValueError(f"Failed to process document: {source_path}")

        base_doc = self._processed_to_document(processed)

        chunks = chunk_documents(
            docs=[base_doc],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"No chunks were produced for document: {source_path}")

        self._enrich_chunk_metadata(chunks, source_path=source_path, processed_path=processed.processed_path)

        chroma_ids = self.chroma.add_chunks(chunks)

        all_chunks = self._append_to_canonical_chunks(chunks)

        self._rebuild_bm25(all_chunks)

        logger.info(
            "Completed ingestion for %s: %d chunks added to Chroma and BM25 rebuilt",
            source_path,
            len(chunks),
        )

        return {
            "source_file": str(source_path),
            "processed_path": processed.processed_path,
            "num_chunks": len(chunks),
            "chroma_ids": chroma_ids,
            "bm25_rebuilt": True,
            "chunks_path": str(self.chunks_path),
            "bm25_index_path": str(self.bm25_index_path),
        }

    def _processed_to_document(self, processed: Any) -> Document:
        """
        Convert a ProcessedDocument into one LangChain Document.
        """
        metadata = dict(processed.metadata or {})
        metadata.update(
            {
                "source": processed.source_path,
                "file_type": processed.file_type,
            }
        )
        if processed.processed_path:
            metadata["processed_path"] = processed.processed_path

        return Document(
            page_content=processed.text,
            metadata=metadata,
        )

    def _append_to_canonical_chunks(self, new_chunks: list[Document]) -> list[Document]:
        """
        Append newly generated chunks to canonical JSONL storage.
        If the file does not exist yet, create it.
        """
        existing_chunks: list[Document] = []
        if self.chunks_path.exists():
            existing_chunks = load_chunks_jsonl(self.chunks_path)

        all_chunks = existing_chunks + new_chunks
        save_chunks_jsonl(all_chunks, self.chunks_path)

        logger.info(
            "Canonical chunk store updated at %s (%d existing + %d new = %d total)",
            self.chunks_path,
            len(existing_chunks),
            len(new_chunks),
            len(all_chunks),
        )
        return all_chunks

    def _rebuild_bm25(self, all_chunks: list[Document]) -> None:
        """
        Rebuild the BM25 artifact from the full canonical chunk set.
        """
        if not all_chunks:
            raise ValueError("Cannot rebuild BM25 with an empty chunk list")

        index = BM25Index.build(all_chunks)
        index.save(self.bm25_index_path)

        logger.info("BM25 index rebuilt and saved to %s", self.bm25_index_path)

    @staticmethod
    def _enrich_chunk_metadata(
        chunks: list[Document],
        source_path: Path,
        processed_path: str | None,
    ) -> None:
        """
        Add stable metadata useful for traceability and deduplication.
        """
        doc_id = DocumentIngestionPipeline._build_doc_id(source_path)

        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}

            chunk.metadata.setdefault("source", str(source_path))
            chunk.metadata.setdefault("filename", source_path.name)
            chunk.metadata.setdefault("doc_id", doc_id)

            if processed_path:
                chunk.metadata.setdefault("processed_path", processed_path)

            chunk.metadata["content_hash"] = hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest()

    @staticmethod
    def _build_doc_id(source_path: Path) -> str:
        """
        Build a stable document identifier from the absolute path.
        """
        return hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()