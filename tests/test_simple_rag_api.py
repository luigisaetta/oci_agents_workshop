"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Unit tests for the FastAPI endpoint exposing the simple RAG agent.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from simple_rag_agent.api import app


def _fake_run_rag_agent(_request: str, vector_store=None) -> dict:
    """Return predictable output for API tests."""
    del vector_store
    return {
        "output": "api answer",
        "retrieved_docs": [{"source": "doc-1", "title": "Doc 1", "page": 1}],
    }


def _fake_build_initialized_vector_store() -> object:
    """Return a dummy pre-indexed store object for startup tests."""
    return object()


def _fake_list_pdf_files(_input_dir: Path) -> list[Path]:
    """Return one dummy PDF path for startup branching tests."""
    return [Path("dummy.pdf")]


def _fake_list_no_pdf_files(_input_dir: Path) -> list[Path]:
    """Return an empty list to force fake knowledge base startup path."""
    return []


def test_invoke_endpoint_returns_output(monkeypatch) -> None:
    """It should return the JSON output produced by the RAG agent."""
    monkeypatch.setattr("simple_rag_agent.api.list_pdf_files", _fake_list_no_pdf_files)
    monkeypatch.setattr(
        "simple_rag_agent.api.build_initialized_vector_store",
        _fake_build_initialized_vector_store,
    )
    monkeypatch.setattr(
        "simple_rag_agent.api.run_rag_agent",
        _fake_run_rag_agent,
    )

    with TestClient(app) as client:
        response = client.post("/invoke", json={"request": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "output": "api answer",
        "retrieved_docs": [{"source": "doc-1", "title": "Doc 1", "page": 1}],
    }


def test_api_startup_uses_pdf_loader_when_pdf_files_exist(monkeypatch) -> None:
    """It should initialize vector store from PDF loader when PDF files exist."""
    pdf_store = object()

    def _fake_run_rag_agent(_request: str, vector_store=None) -> dict:
        assert vector_store is pdf_store
        return {
            "output": "from pdf store",
            "retrieved_docs": [{"source": "doc-pdf", "text": "context"}],
        }

    def _fake_pdf_vector_store_builder(
        input_dir: Path, chunk_size=800, chunk_overlap=200
    ):
        del input_dir, chunk_size, chunk_overlap
        return pdf_store

    monkeypatch.setattr("simple_rag_agent.api.list_pdf_files", _fake_list_pdf_files)
    monkeypatch.setattr(
        "simple_rag_agent.api.build_pdf_vector_store",
        _fake_pdf_vector_store_builder,
    )
    monkeypatch.setattr(
        "simple_rag_agent.api.run_rag_agent",
        _fake_run_rag_agent,
    )

    with TestClient(app) as client:
        response = client.post("/invoke", json={"request": "hello"})

    assert response.status_code == 200
    assert response.json()["output"] == "from pdf store"
