"""Semantic top-k retrieval with a transparent cosine relevance threshold."""
from __future__ import annotations
from typing import Any


class PolicyRetriever:
    def __init__(self, vector_store: Any, embedder: Any, relevance_threshold: float = 0.25):
        if not -1 <= relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between -1 and 1")
        self.vector_store, self.embedder = vector_store, embedder
        self.relevance_threshold = relevance_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return only the highest-scoring chunks that meet the configured threshold."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        vector = self.embedder.embed(query, "RETRIEVAL_QUERY")
        return [item for item in self.vector_store.search(vector, top_k=top_k)
                if item["score"] >= self.relevance_threshold]
