"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: In-memory fake vector store and semantic search helpers for RAG examples.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from langchain_core.documents import Document

DEFAULT_DOCUMENTS: Sequence[Document] = (
    Document(
        page_content=(
            "OCI Generative AI provides foundation models and inference endpoints."
        ),
        metadata={"source": "doc-1"},
    ),
    Document(
        page_content=(
            "Dedicated AI Clusters can be used for enterprise-grade model serving in OCI."
        ),
        metadata={"source": "doc-2"},
    ),
    Document(
        page_content=(
            "LangChain and LangGraph can orchestrate multi-step AI workflows."
        ),
        metadata={"source": "doc-3"},
    ),
    Document(
        page_content=(
            "OCI Identity and Access Management controls access with groups and policies."
        ),
        metadata={"source": "doc-4"},
    ),
    Document(
        page_content=(
            "OCI Object Storage can be used to store unstructured data for AI workloads."
        ),
        metadata={"source": "doc-5"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation combines retrieval with model generation."
        ),
        metadata={"source": "doc-6"},
    ),
    Document(
        page_content=(
            "Embeddings map text into vectors that can be compared by similarity."
        ),
        metadata={"source": "doc-7"},
    ),
    Document(
        page_content=("FastAPI can expose AI workflows through simple HTTP endpoints."),
        metadata={"source": "doc-8"},
    ),
    Document(
        page_content=(
            "Uvicorn is a common ASGI server used to run FastAPI applications."
        ),
        metadata={"source": "doc-9"},
    ),
    Document(
        page_content=(
            "A minimal RAG pipeline can be built with semantic search and answer generation."
        ),
        metadata={"source": "doc-10"},
    ),
)

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

    def search(self, query: str, embedding_client: Any) -> List[Document]:
        """Return top-k documents semantically closest to the query."""
        query_vector = embedding_client.embed_query(query)
        doc_vectors = embedding_client.embed_documents(
            [document.page_content for document in self._base_documents]
        )

        ranked_pairs = []
        for document, vector in zip(self._base_documents, doc_vectors, strict=True):
            score = _cosine_similarity(query_vector, vector)
            ranked_pairs.append((score, document))

        ranked_pairs.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked_pairs[: self._top_k]]
