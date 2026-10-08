"""Persistent local Chroma vector store using externally generated embeddings."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from .chunker import DocumentChunk


class ChromaVectorStore:
    """Store policy chunks and vectors in a local persistent Chroma collection."""
    def __init__(self, path: str | Path = "data/processed/policy_chroma", collection_name: str = "financial_policies"):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("Chroma is required for the local vector store; install requirements.txt") from exc
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})

    @property
    def count(self) -> int:
        return int(self.collection.count())

    def upsert(self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace document chunks with precomputed Gemini vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk needs exactly one embedding")
        if not chunks:
            return
        metadata = []
        for chunk in chunks:
            # Chroma metadata values must be scalar and non-null.
            metadata.append({key: value for key, value in chunk.metadata.items()
                             if value is not None and isinstance(value, (str, int, float, bool))})
        self.collection.upsert(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
                               metadatas=metadata, embeddings=[list(map(float, v)) for v in embeddings])

    def replace_all(self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Synchronize the index to the current knowledge-base files, removing stale chunks."""
        if self.count:
            old_ids = self.collection.get(include=[]).get("ids", [])
            if old_ids:
                self.collection.delete(ids=old_ids)
        self.upsert(chunks, embeddings)

    def get_existing_embeddings(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        """Read stored vectors for unchanged chunks so document embeddings are not billed twice."""
        if not chunk_ids or not self.count:
            return {}
        result = self.collection.get(ids=list(chunk_ids), include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            return {}
        return {chunk_id: list(map(float, embeddings[index])) for index, chunk_id in enumerate(result["ids"])}

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Return nearest chunks with normalized cosine relevance scores."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self.count:
            return []
        raw = self.collection.query(query_embeddings=[list(map(float, embedding))], n_results=min(top_k, self.count),
                                     include=["documents", "metadatas", "distances"])
        results = []
        for i, text in enumerate(raw["documents"][0]):
            distance = float(raw["distances"][0][i])
            results.append({"id": raw["ids"][0][i], "text": text,
                            "metadata": raw["metadatas"][0][i] or {}, "score": max(-1.0, min(1.0, 1 - distance))})
        return results
