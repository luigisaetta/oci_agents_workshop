"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Unit tests for PDF loading, chunking, and embedding pipeline.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods,missing-function-docstring

from pathlib import Path

import pytest
from langchain_core.documents import Document

from simple_rag_agent import pdf_loader


class _FakeEmbeddingClient:
    """Fake embedding client for deterministic vector generation."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic vector for each input text."""
        return [[float(index), 0.5, 1.5] for index, _ in enumerate(texts, start=1)]

    def embed_query(self, query: str) -> list[float]:
        """Return a fixed query vector."""
        if not query:
            raise ValueError("Query must not be empty.")
        return [1.0, 0.0, 0.0]


def test_load_pdf_documents_sets_source_title_and_page(
    monkeypatch, tmp_path: Path
) -> None:
    """It should map PDF pages to documents with source/title/page metadata."""

    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeMetadata:
        title = "My PDF Title"

    class _FakeReader:
        def __init__(self, _path: str) -> None:
            self.metadata = _FakeMetadata()
            self.pages = [_FakePage("Page one"), _FakePage("Page two")]

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder")

    monkeypatch.setattr(pdf_loader, "PdfReader", _FakeReader)

    documents = pdf_loader.load_pdf_documents(tmp_path)

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "sample.pdf"
    assert documents[0].metadata["title"] == "My PDF Title"
    assert documents[0].metadata["page"] == 1


def test_chunk_documents_splits_long_text() -> None:
    """It should split long documents according to chunking settings."""
    documents = [Document(page_content="A" * 1200, metadata={"source": "x.pdf"})]

    chunks = pdf_loader.chunk_documents(documents, chunk_size=400, chunk_overlap=100)

    assert len(chunks) > 1


def test_add_chunks_to_vector_store_returns_added_count() -> None:
    """It should add all chunks to vector store and return total added count."""
    vector_store = pdf_loader.InMemoryVectorStore(embedding=_FakeEmbeddingClient())
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "x.pdf"}),
        Document(page_content="chunk 2", metadata={"source": "x.pdf"}),
    ]

    added_count = pdf_loader.add_chunks_to_vector_store(
        chunks,
        vector_store=vector_store,
        batch_size=1,
    )

    assert added_count == 2


def test_build_pdf_vector_store_requires_pdf_files(tmp_path: Path) -> None:
    """It should fail when there are no PDF files in input directory."""
    with pytest.raises(ValueError, match="No PDF files"):
        pdf_loader.load_pdf_documents(tmp_path)


def test_build_pdf_vector_store_creates_index(monkeypatch, tmp_path: Path) -> None:
    """It should create an indexed vector store from loaded and chunked docs."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")
    monkeypatch.setenv("SIMPLE_RAG_TOP_K", "2")
    monkeypatch.setattr(
        pdf_loader,
        "load_pdf_documents",
        lambda _input_dir: [
            Document(page_content="A" * 1000, metadata={"source": "x.pdf"})
        ],
    )
    monkeypatch.setattr(
        pdf_loader,
        "build_embedding_client",
        lambda _runtime_config: _FakeEmbeddingClient(),
    )

    store = pdf_loader.build_pdf_vector_store(
        input_dir=tmp_path,
        chunk_size=300,
        chunk_overlap=50,
    )

    results = store.similarity_search("query", k=2)
    assert len(results) == 2
