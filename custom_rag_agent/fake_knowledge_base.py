"""
Author: L. Saetta
Date last modified: 2026-04-18
License: MIT
Description: Fake knowledge base documents used by the simple in-memory RAG example.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

FAKE_KNOWLEDGE_BASE_TEXTS = [
    "OCI Generative AI provides foundation models and inference endpoints.",
    "Dedicated AI Clusters can be used for enterprise-grade model serving in OCI.",
    "LangChain and LangGraph can orchestrate multi-step AI workflows.",
    "OCI Identity and Access Management controls access with groups and policies.",
    "OCI Object Storage can be used to store unstructured data for AI workloads.",
    "Retrieval-Augmented Generation combines retrieval with model generation.",
    "Embeddings map text into vectors that can be compared by similarity.",
    "FastAPI can expose AI workflows through simple HTTP endpoints.",
    "Uvicorn is a common ASGI server used to run FastAPI applications.",
    "A minimal RAG pipeline can be built with semantic search and answer generation.",
]


def build_fake_documents() -> List[Document]:
    """Build LangChain documents from static knowledge base strings.

    Returns:
        List[Document]: Documents with page content and source metadata.
    """
    documents: List[Document] = []
    for index, text in enumerate(FAKE_KNOWLEDGE_BASE_TEXTS, start=1):
        documents.append(
            Document(page_content=text, metadata={"source": f"doc-{index}"})
        )
    return documents
