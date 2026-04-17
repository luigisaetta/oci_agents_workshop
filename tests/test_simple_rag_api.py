"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Unit tests for the FastAPI endpoint exposing the simple RAG agent.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from simple_rag_agent.api import app


def _fake_run_rag_agent(_request: str, vector_store=None) -> dict:
    """Return predictable output for API tests."""
    del vector_store
    return {
        "output": "api answer",
        "retrieved_docs": [{"source": "doc-1", "text": "context"}],
    }


def _fake_build_initialized_vector_store() -> object:
    """Return a dummy pre-indexed store object for startup tests."""
    return object()


def test_invoke_endpoint_returns_output(monkeypatch) -> None:
    """It should return the JSON output produced by the RAG agent."""
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
        "retrieved_docs": [{"source": "doc-1", "text": "context"}],
    }
