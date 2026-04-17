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

from simple_rag_agent import rag_agent


class _FakeEmbeddingClient:
    """Fake embedding client with deterministic vectors."""

    def embed_query(self, query: str) -> list[float]:
        """Return a query vector aligned with first dimension."""
        if not query:
            raise ValueError("Query must not be empty.")
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors based on document index."""
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        return vectors[: len(texts)]


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

    step = rag_agent.SemanticSearcher(base_documents=documents)
    result = step.invoke({"user_input": "test query"})

    assert len(result["documents"]) == 3
    assert result["documents"][0].page_content == "d1"


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


def test_run_rag_agent_returns_json_output(monkeypatch) -> None:
    """It should run the graph and return output in JSON-compatible format."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
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

    assert result == {"output": "fake answer"}
