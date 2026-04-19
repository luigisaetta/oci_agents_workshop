"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Unit tests for the FastAPI endpoint exposing the custom RAG agent.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from custom_rag_agent.api import app


def _fake_run_rag_agent(
    _request: str, history=None, vector_store=None, top_k=None
) -> dict:
    """Return predictable output for API tests."""
    assert history == []
    assert top_k is None
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
    monkeypatch.setattr("custom_rag_agent.api.list_pdf_files", _fake_list_no_pdf_files)
    monkeypatch.setattr(
        "custom_rag_agent.api.build_initialized_vector_store",
        _fake_build_initialized_vector_store,
    )
    monkeypatch.setattr(
        "custom_rag_agent.api.run_rag_agent",
        _fake_run_rag_agent,
    )

    with TestClient(app) as client:
        response = client.post("/invoke", json={"request": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "output": "api answer",
        "retrieved_docs": [{"source": "doc-1", "title": "Doc 1", "page": 1}],
    }


def test_invoke_endpoint_forwards_history(monkeypatch) -> None:
    """It should forward history messages from payload to the RAG agent."""
    monkeypatch.setattr("custom_rag_agent.api.list_pdf_files", _fake_list_no_pdf_files)
    monkeypatch.setattr(
        "custom_rag_agent.api.build_initialized_vector_store",
        _fake_build_initialized_vector_store,
    )

    def _fake_run_with_history(
        _request: str, history=None, vector_store=None, top_k=None
    ) -> dict:
        assert history == [{"role": "user", "content": "Previous question"}]
        assert top_k is None
        del vector_store
        return {"output": "ok", "retrieved_docs": []}

    monkeypatch.setattr(
        "custom_rag_agent.api.run_rag_agent",
        _fake_run_with_history,
    )

    with TestClient(app) as client:
        response = client.post(
            "/invoke",
            json={
                "request": "follow-up",
                "history": [{"role": "user", "content": "Previous question"}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"output": "ok", "retrieved_docs": []}


def test_invoke_endpoint_forwards_top_k(monkeypatch) -> None:
    """It should forward top_k from payload to the RAG agent."""
    monkeypatch.setattr("custom_rag_agent.api.list_pdf_files", _fake_list_no_pdf_files)
    monkeypatch.setattr(
        "custom_rag_agent.api.build_initialized_vector_store",
        _fake_build_initialized_vector_store,
    )

    def _fake_run_with_top_k(
        _request: str, history=None, vector_store=None, top_k=None
    ):
        assert history == []
        assert top_k == 7
        del vector_store
        return {"output": "ok", "retrieved_docs": []}

    monkeypatch.setattr(
        "custom_rag_agent.api.run_rag_agent",
        _fake_run_with_top_k,
    )

    with TestClient(app) as client:
        response = client.post(
            "/invoke",
            json={"request": "follow-up", "top_k": 7},
        )

    assert response.status_code == 200
    assert response.json() == {"output": "ok", "retrieved_docs": []}


def test_api_startup_uses_pdf_loader_when_pdf_files_exist(monkeypatch) -> None:
    """It should initialize vector store from PDF loader when PDF files exist."""
    pdf_store = object()

    def _fake_run_rag_agent(
        _request: str, history=None, vector_store=None, top_k=None
    ) -> dict:
        assert history == []
        assert vector_store is pdf_store
        assert top_k is None
        return {
            "output": "from pdf store",
            "retrieved_docs": [{"source": "doc-pdf", "text": "context"}],
        }

    def _fake_pdf_vector_store_builder(
        input_dir: Path, chunk_size=800, chunk_overlap=200
    ):
        del input_dir, chunk_size, chunk_overlap
        return pdf_store

    monkeypatch.setattr("custom_rag_agent.api.list_pdf_files", _fake_list_pdf_files)
    monkeypatch.setattr(
        "custom_rag_agent.api.build_pdf_vector_store",
        _fake_pdf_vector_store_builder,
    )
    monkeypatch.setattr(
        "custom_rag_agent.api.run_rag_agent",
        _fake_run_rag_agent,
    )

    with TestClient(app) as client:
        response = client.post("/invoke", json={"request": "hello"})

    assert response.status_code == 200
    assert response.json()["output"] == "from pdf store"


def test_invoke_stream_endpoint_returns_sse_events(monkeypatch) -> None:
    """It should expose SSE events for progressive streaming invocations."""
    monkeypatch.setattr("custom_rag_agent.api.list_pdf_files", _fake_list_no_pdf_files)
    monkeypatch.setattr(
        "custom_rag_agent.api.build_initialized_vector_store",
        _fake_build_initialized_vector_store,
    )

    def _fake_stream_events(_request: str, history=None, vector_store=None, top_k=None):
        assert history == []
        assert vector_store is not None
        assert top_k is None
        yield {"event": "step_started", "step": "query_rewriter"}
        yield {"event": "retrieval_results", "retrieved_docs": [{"source": "doc-1"}]}
        yield {"event": "final_answer_token", "token": "hello"}
        yield {"event": "completed", "output": "hello", "retrieved_docs": []}

    monkeypatch.setattr(
        "custom_rag_agent.api.stream_rag_agent_events",
        _fake_stream_events,
    )

    with TestClient(app) as client:
        response = client.post("/invoke/stream", json={"request": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    payload_lines = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    events = [json.loads(line) for line in payload_lines]

    assert events[0]["event"] == "step_started"
    assert events[1]["event"] == "retrieval_results"
    assert events[2]["event"] == "final_answer_token"
    assert events[-1]["event"] == "completed"


def test_invoke_stream_endpoint_forwards_top_k(monkeypatch) -> None:
    """It should forward top_k from payload to streaming RAG events."""
    monkeypatch.setattr("custom_rag_agent.api.list_pdf_files", _fake_list_no_pdf_files)
    monkeypatch.setattr(
        "custom_rag_agent.api.build_initialized_vector_store",
        _fake_build_initialized_vector_store,
    )

    def _fake_stream_events(_request: str, history=None, vector_store=None, top_k=None):
        assert history == []
        assert vector_store is not None
        assert top_k == 3
        yield {"event": "completed", "output": "ok", "retrieved_docs": []}

    monkeypatch.setattr(
        "custom_rag_agent.api.stream_rag_agent_events",
        _fake_stream_events,
    )

    with TestClient(app) as client:
        response = client.post(
            "/invoke/stream",
            json={"request": "hello", "top_k": 3},
        )

    assert response.status_code == 200
    payload_lines = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    events = [json.loads(line) for line in payload_lines]
    assert events[-1]["event"] == "completed"
