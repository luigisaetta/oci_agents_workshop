"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: In-memory fake vector store and semantic search helpers for RAG examples.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from langchain_core.documents import Document

from simple_rag_agent.fake_knowledge_base import build_fake_documents

DEFAULT_DOCUMENTS: Sequence[Document] = tuple(build_fake_documents())

TOP_K_RESULTS = 4


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    numerator = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    norm_a = sum(value * value for value in vector_a) ** 0.5
    norm_b = sum(value * value for value in vector_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


# pylint: disable=too-few-public-methods
class InMemoryVectorStore:
    """Simple in-memory vector store using OCI embeddings for semantic search."""

    def __init__(
        self,
        base_documents: Sequence[Document] = DEFAULT_DOCUMENTS,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        self._base_documents = list(base_documents)
        self._top_k = top_k
        self._indexed_documents: List[Document] = []
        self._document_vectors: List[Sequence[float]] = []

    def index(self, embedding_client: Any) -> None:
        """Build in-memory vectors for all base documents."""
        self._indexed_documents = list(self._base_documents)
        self._document_vectors = embedding_client.embed_documents(
            [document.page_content for document in self._indexed_documents]
        )

    def search(self, query: str, embedding_client: Any) -> List[Document]:
        """Return top-k documents semantically closest to the query."""
        if not self._indexed_documents or not self._document_vectors:
            raise RuntimeError("Vector store is not indexed. Call index() first.")

        query_vector = embedding_client.embed_query(query)

        ranked_pairs = []
        for document, vector in zip(
            self._indexed_documents, self._document_vectors, strict=True
        ):
            score = _cosine_similarity(query_vector, vector)
            ranked_pairs.append((score, document))

        ranked_pairs.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked_pairs[: self._top_k]]
