"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Unit tests for the simple two-step LangGraph RAG agent.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods,duplicate-code

from typing import Any

import pytest
from langchain_core.documents import Document

from custom_rag_agent import rag_agent


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


class _FakeStreamLlm:
    """Fake LLM returning deterministic streaming tokens."""

    def stream(self, _prompt: str):
        """Yield a deterministic token stream."""
        yield "fake "
        yield "stream"


class _FakeRewriteLlm:
    """Fake LLM used by query rewrite tests."""

    def invoke(self, prompt: str) -> Any:
        """Return deterministic rewritten query."""
        if "Standalone search query:" not in prompt:
            raise ValueError("Rewrite prompt was not provided.")
        return "rewritten standalone query"


class _FakeVerboseRewriteLlm:
    """Fake LLM returning extra narrative after the standalone query."""

    def invoke(self, _prompt: str) -> Any:
        """Return verbose rewrite output with explanation text."""
        return (
            'Standalone search query: "Is Oracle Open Agent Spec open source?"\n'
            "Context-complete explanation: Additional details not for search."
        )


class _FakeVectorStore:
    """Minimal vector store implementing only similarity_search."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.last_query = ""

    def similarity_search(
        self, query: str, k: int = 4, **_kwargs: Any
    ) -> list[Document]:
        """Return first k documents in deterministic order."""
        self.last_query = query
        return self._docs[:k]


class _ReusableVectorStore:
    """Store stub used to verify no fake-seeding side effects occur."""

    def add_documents(self, _documents: list[Document]) -> list[str]:
        """Fail if fake seeding tries to mutate this external store."""
        raise AssertionError("External vector store must not be seeded with fake docs.")


class _FakeRetrievalGraph:
    """Graph stub returning deterministic LangGraph update events."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def stream(self, _state: dict[str, Any], stream_mode: str = "updates"):
        """Yield two graph updates in the same shape returned by LangGraph."""
        assert stream_mode == "updates"
        yield {
            "query_rewriter": {
                "search_query": "rewritten standalone query",
                "runtime_config": {
                    "OCI_MODEL_ID": "model-id",
                    "OCI_REGION": "us-chicago-1",
                    "OCI_SERVICE_ENDPOINT": "https://example.test",
                    "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
                    "OCI_PROVIDER": "generic",
                    "OCI_AUTH_TYPE": "API_KEY",
                    "OCI_AUTH_PROFILE": "DEFAULT",
                    "OCI_EMBED_MODEL_ID": "cohere.embed-english-v3.0",
                    "SIMPLE_RAG_TOP_K": "4",
                },
            }
        }
        yield {
            "semantic_searcher": {
                "documents": self._docs,
                "runtime_config": {
                    "OCI_MODEL_ID": "model-id",
                    "OCI_REGION": "us-chicago-1",
                    "OCI_SERVICE_ENDPOINT": "https://example.test",
                    "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
                    "OCI_PROVIDER": "generic",
                    "OCI_AUTH_TYPE": "API_KEY",
                    "OCI_AUTH_PROFILE": "DEFAULT",
                    "OCI_EMBED_MODEL_ID": "cohere.embed-english-v3.0",
                    "SIMPLE_RAG_TOP_K": "4",
                },
            }
        }


def test_build_initialized_vector_store_reuses_external_store_as_is(
    monkeypatch,
) -> None:
    """It should reuse external vector stores without loading fake KB documents."""
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    monkeypatch.delenv("OCI_EMBED_MODEL_ID", raising=False)

    external_store = _ReusableVectorStore()
    reused_store = rag_agent.build_initialized_vector_store(vector_store=external_store)

    assert reused_store is external_store


def test_semantic_searcher_returns_top_documents(monkeypatch) -> None:
    """It should return ranked documents from semantic search."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")

    documents = [
        Document(page_content="d1", metadata={"source": "a"}),
        Document(page_content="d2", metadata={"source": "b"}),
        Document(page_content="d3", metadata={"source": "c"}),
    ]

    step = rag_agent.SemanticSearcher(vector_store=_FakeVectorStore(documents))
    result = step.invoke({"user_input": "test query"})

    assert len(result["documents"]) == 3
    assert result["documents"][0].page_content == "d1"


def test_semantic_searcher_uses_rewritten_search_query(monkeypatch) -> None:
    """It should use search_query when available in state."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")

    vector_store = _FakeVectorStore(
        [Document(page_content="d1", metadata={"source": "a"})]
    )
    step = rag_agent.SemanticSearcher(vector_store=vector_store)
    step.invoke(
        {
            "user_input": "original query",
            "search_query": "rewritten standalone query",
        }
    )

    assert vector_store.last_query == "rewritten standalone query"


def test_semantic_searcher_applies_top_k_filter(monkeypatch) -> None:
    """It should limit retrieved documents to the configured top_k value."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setenv("SIMPLE_RAG_TOP_K", "2")

    documents = [
        Document(page_content=f"doc-{index}", metadata={"source": f"s-{index}"})
        for index in range(1, 7)
    ]

    step = rag_agent.SemanticSearcher(vector_store=_FakeVectorStore(documents))
    result = step.invoke({"user_input": "test query"})

    assert len(result["documents"]) == 2
    assert result["documents"][0].page_content == "doc-1"


def test_query_rewriter_keeps_original_query_when_history_is_empty(
    monkeypatch,
) -> None:
    """It should bypass rewrite when history is empty."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setattr(
        rag_agent,
        "build_llm",
        lambda _runtime_config: (_ for _ in ()).throw(
            AssertionError("LLM must not be called for empty history.")
        ),
    )

    step = rag_agent.QueryRewriter()
    result = step.invoke({"user_input": "original query", "history": []})

    assert result["search_query"] == "original query"


def test_query_rewriter_rewrites_query_when_history_is_present(monkeypatch) -> None:
    """It should produce a rewritten standalone query when history is provided."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setattr(
        rag_agent,
        "build_llm",
        lambda _runtime_config: _FakeRewriteLlm(),
    )

    step = rag_agent.QueryRewriter()
    result = step.invoke(
        {
            "user_input": "What about pricing?",
            "history": [
                {"role": "user", "content": "Tell me about Dedicated AI Cluster"},
                {"role": "assistant", "content": "It is OCI managed infrastructure."},
            ],
        }
    )

    assert result["search_query"] == "rewritten standalone query"


def test_query_rewriter_strips_explanation_from_verbose_output(monkeypatch) -> None:
    """It should keep only the standalone query when model adds extra text."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setattr(
        rag_agent,
        "build_llm",
        lambda _runtime_config: _FakeVerboseRewriteLlm(),
    )

    step = rag_agent.QueryRewriter()
    result = step.invoke(
        {
            "user_input": "Is it open source?",
            "history": [{"role": "user", "content": "Oracle Open Agent Spec status?"}],
        }
    )

    assert result["search_query"] == "Is Oracle Open Agent Spec open source?"


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
    assert result["retrieved_docs"] == [{"source": "x"}]


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

    with pytest.raises(ValueError, match="SIMPLE_RAG_TOP_K"):
        rag_agent.SemanticSearcher(vector_store=_FakeVectorStore([])).invoke(
            {"user_input": "test query"}
        )


def test_stream_rag_agent_events_emits_progress_retrieval_and_tokens(
    monkeypatch,
) -> None:
    """It should stream step updates, retrieval metadata, and answer tokens."""
    docs = [Document(page_content="context text", metadata={"source": "doc-1"})]

    monkeypatch.setattr(
        rag_agent,
        "build_retrieval_graph",
        lambda vector_store=None: _FakeRetrievalGraph(docs),
    )
    monkeypatch.setattr(
        rag_agent,
        "build_llm",
        lambda _runtime_config: _FakeStreamLlm(),
    )

    events = list(
        rag_agent.stream_rag_agent_events(
            "Explain DAC",
            history=[{"role": "user", "content": "Previous"}],
        )
    )

    event_types = [event["event"] for event in events]
    assert "step_started" in event_types
    assert "step_completed" in event_types
    assert "retrieval_results" in event_types
    assert "final_answer_token" in event_types
    assert event_types[-1] == "completed"

    completed_event = events[-1]
    assert completed_event["output"] == "fake stream"
    assert completed_event["retrieved_docs"] == [{"source": "doc-1"}]
