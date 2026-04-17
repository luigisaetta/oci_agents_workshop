"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Unit tests for the fake knowledge base used by the simple RAG agent.
"""

from __future__ import annotations

from simple_rag_agent.fake_knowledge_base import build_fake_documents


def test_build_fake_documents_returns_expected_defaults() -> None:
    """It should build 10 default documents with stable source metadata."""
    documents = build_fake_documents()

    assert len(documents) == 10
    assert documents[0].metadata["source"] == "doc-1"
    assert documents[-1].metadata["source"] == "doc-10"
