"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Unit tests for quickstart vector store semantic search example.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document

from quickstart import vector_store_search01


class _FakeVectorStores:
    """Fake vector_stores namespace exposing a deterministic search method."""

    def __init__(self, result_items: list[Any]) -> None:
        self._result_items = result_items
        self.last_search_kwargs: dict[str, Any] = {}

    def search(self, **kwargs: Any) -> Any:
        """Return deterministic search payload and save call arguments."""
        self.last_search_kwargs = kwargs
        return SimpleNamespace(data=self._result_items)


class _FakeClient:
    """Fake OpenAI-compatible client with vector_stores namespace."""

    def __init__(self, result_items: list[Any]) -> None:
        self.vector_stores = _FakeVectorStores(result_items)


def test_collect_vector_store_runtime_config_requires_vector_store_vars(
    monkeypatch,
) -> None:
    """It should fail when vector store-specific env vars are missing."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.delenv("OCI_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OCI_VECTOR_STORE_ID", raising=False)
    monkeypatch.delenv("OCI_OPENAI_PROJECT_ID", raising=False)

    with pytest.raises(ValueError, match="OCI_OPENAI_BASE_URL"):
        vector_store_search01.collect_vector_store_runtime_config()


def test_semantic_search_vector_store_returns_sorted_langchain_documents() -> None:
    """It should sort by score descending and return LangChain documents."""
    low_score = SimpleNamespace(
        score=0.3,
        filename="low.pdf",
        file_id="file-low",
        content=[SimpleNamespace(text="low chunk")],
        additional_properties={"page_numbers": [4], "chunk_id": "low-1"},
    )
    high_score = SimpleNamespace(
        score=0.9,
        filename="high.pdf",
        file_id="file-high",
        content=[{"text": "high chunk"}],
        additional_properties={"page_numbers": [2], "chunk_id": "high-1"},
    )
    client = _FakeClient([low_score, high_score])

    documents = vector_store_search01.semantic_search_vector_store(
        query="Find best chunk",
        vector_store_id="vs_123",
        project_id="proj_123",
        client=client,
        max_num_results=5,
    )

    assert isinstance(documents[0], Document)
    assert documents[0].metadata["source"] == "high.pdf"
    assert documents[0].metadata["page"] == 2
    assert documents[0].page_content == "high chunk"
    assert client.vector_stores.last_search_kwargs["extra_headers"] == {
        "OpenAI-Project": "proj_123"
    }


def test_build_retrieved_docs_metadata_matches_custom_rag_shape() -> None:
    """It should return metadata-only dictionaries like custom_rag_agent."""
    documents = [
        Document(page_content="chunk", metadata={"source": "a.pdf", "title": "A"})
    ]

    retrieved_docs = vector_store_search01.build_retrieved_docs_metadata(documents)

    assert retrieved_docs == [{"source": "a.pdf", "title": "A"}]


def test_semantic_search_vector_store_rejects_invalid_inputs() -> None:
    """It should fail for empty query or non-positive max_num_results."""
    client = _FakeClient([])

    with pytest.raises(ValueError, match="Query must not be empty"):
        vector_store_search01.semantic_search_vector_store(
            query=" ",
            vector_store_id="vs_123",
            project_id="proj_123",
            client=client,
        )

    with pytest.raises(ValueError, match="max_num_results"):
        vector_store_search01.semantic_search_vector_store(
            query="hello",
            vector_store_id="vs_123",
            project_id="proj_123",
            client=client,
            max_num_results=0,
        )


def test_main_uses_query_from_cli_input(monkeypatch, capsys) -> None:
    """It should read query from CLI input instead of environment variables."""
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        vector_store_search01,
        "collect_vector_store_runtime_config",
        lambda: {
            "OCI_OPENAI_BASE_URL": "https://example.test/v1",
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
            "OCI_AUTH_PROFILE": "DEFAULT",
            "OCI_VECTOR_STORE_ID": "vs_123",
            "OCI_OPENAI_PROJECT_ID": "proj_123",
        },
    )
    monkeypatch.setattr(
        vector_store_search01,
        "print_oci_runtime_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        vector_store_search01,
        "build_oci_openai_client",
        lambda **_kwargs: object(),
    )

    def _fake_search(**kwargs: Any) -> list[Document]:
        captured.update(kwargs)
        return [Document(page_content="chunk", metadata={"source": "doc-1"})]

    monkeypatch.setattr(
        vector_store_search01,
        "semantic_search_vector_store",
        _fake_search,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["quickstart/vector_store_search01.py", "query from cli"],
    )

    vector_store_search01.main()
    assert captured["query"] == "query from cli"

    printed = capsys.readouterr().out
    assert "Retrieved Docs Metadata" in printed
