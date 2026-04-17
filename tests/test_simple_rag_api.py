"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Unit tests for the FastAPI endpoint exposing the simple RAG agent.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from simple_rag_agent.api import app


def test_invoke_endpoint_returns_output(monkeypatch) -> None:
    """It should return the JSON output produced by the RAG agent."""
    monkeypatch.setattr(
        "simple_rag_agent.api.run_rag_agent",
        lambda _request: {
            "output": "api answer",
            "retrieved_docs": [{"source": "doc-1", "text": "context"}],
        },
    )

    client = TestClient(app)
    response = client.post("/invoke", json={"request": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "output": "api answer",
        "retrieved_docs": [{"source": "doc-1", "text": "context"}],
    }
