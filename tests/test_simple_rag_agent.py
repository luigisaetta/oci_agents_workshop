"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Unit tests for the simple two-step LangGraph RAG agent.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods

from typing import Any

from langchain_core.documents import Document
import pytest

from simple_rag_agent import rag_agent
from simple_rag_agent.in_memory_vector_store import InMemoryVectorStore


class _FakeEmbeddingClient:
    """Fake embedding client with deterministic vectors."""

    def embed_query(self, query: str) -> list[float]:
        """Return a query vector aligned with first dimension."""
        if not query:
            raise ValueError("Query must not be empty.")
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors based on document index."""
        vectors = []
        for index, _ in enumerate(texts, start=1):
            vectors.append([1.0 / index, 0.0, 0.0])
        return vectors


class _FakeLlm:
    """Fake LLM returning predictable text output."""

    def invoke(self, prompt: str) -> Any:
        """Return fake model response while checking prompt is populated."""
        if "Context:" not in prompt:
            raise ValueError("Context was not provided in prompt.")
        return "fake answer"


def test_semantic_searcher_returns_top_documents(monkeypatch) -> None:
    """It should return ranked documents from semantic search."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setattr(
        rag_agent,
        "build_embedding_client",
        lambda _runtime_config: _FakeEmbeddingClient(),
    )

    documents = [
        Document(page_content="d1", metadata={"source": "a"}),
        Document(page_content="d2", metadata={"source": "b"}),
        Document(page_content="d3", metadata={"source": "c"}),
    ]

    vector_store = InMemoryVectorStore(base_documents=documents, top_k=4)
    vector_store.index(_FakeEmbeddingClient())
    step = rag_agent.SemanticSearcher(vector_store=vector_store)
    result = step.invoke({"user_input": "test query"})

    assert len(result["documents"]) == 3
    assert result["documents"][0].page_content == "d1"


def test_semantic_searcher_applies_top_k_filter(monkeypatch) -> None:
    """It should limit retrieved documents to the configured top_k value."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setattr(
        rag_agent,
        "build_embedding_client",
        lambda _runtime_config: _FakeEmbeddingClient(),
    )

    documents = [
        Document(page_content=f"doc-{index}", metadata={"source": f"s-{index}"})
        for index in range(1, 7)
    ]

    vector_store = InMemoryVectorStore(base_documents=documents, top_k=4)
    vector_store.index(_FakeEmbeddingClient())
    step = rag_agent.SemanticSearcher(vector_store=vector_store)
    result = step.invoke({"user_input": "test query"})

    assert len(result["documents"]) == 4
    assert result["documents"][0].page_content == "doc-1"


def test_semantic_searcher_uses_top_k_from_env(monkeypatch) -> None:
    """It should use SIMPLE_RAG_TOP_K when default vector store is used."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setenv("SIMPLE_RAG_TOP_K", "2")
    monkeypatch.setattr(
        rag_agent,
        "build_embedding_client",
        lambda _runtime_config: _FakeEmbeddingClient(),
    )

    step = rag_agent.SemanticSearcher()
    result = step.invoke({"user_input": "test query"})

    assert len(result["documents"]) == 2


def test_answer_generator_returns_output(monkeypatch) -> None:
    """It should create output text from retrieved documents."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setattr(
        rag_agent,
        "build_llm",
        lambda _runtime_config: _FakeLlm(),
    )

    step = rag_agent.AnswerGenerator()
    state = {
        "user_input": "What is OCI?",
        "documents": [Document(page_content="OCI info", metadata={"source": "x"})],
    }

    result = step.invoke(state)
    assert result["output"] == "fake answer"
    assert result["retrieved_docs"] == [{"source": "x", "text": "OCI info"}]


def test_run_rag_agent_returns_json_output(monkeypatch) -> None:
    """It should run the graph and return output in JSON-compatible format."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setenv("SIMPLE_RAG_TOP_K", "2")
    monkeypatch.setattr(
        rag_agent,
        "build_embedding_client",
        lambda _runtime_config: _FakeEmbeddingClient(),
    )
    monkeypatch.setattr(
        rag_agent,
        "build_llm",
        lambda _runtime_config: _FakeLlm(),
    )

    result = rag_agent.run_rag_agent("Explain DAC")

    assert result["output"] == "fake answer"
    assert len(result["retrieved_docs"]) == 2


def test_collect_rag_runtime_config_requires_positive_integer_top_k(
    monkeypatch,
) -> None:
    """It should fail when SIMPLE_RAG_TOP_K is not a positive integer."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setenv("SIMPLE_RAG_TOP_K", "zero")
    monkeypatch.setattr(
        rag_agent,
        "build_embedding_client",
        lambda _runtime_config: _FakeEmbeddingClient(),
    )

    with pytest.raises(ValueError, match="SIMPLE_RAG_TOP_K"):
        rag_agent.SemanticSearcher().invoke({"user_input": "test query"})
