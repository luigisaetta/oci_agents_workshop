"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Unit tests for custom RAG CLI client output formatting.
"""

from __future__ import annotations

from custom_rag_agent.client import format_response


def test_format_response_includes_source_title_and_page() -> None:
    """It should render source, title, and page for retrieved metadata."""
    payload = {
        "output": "Answer text",
        "retrieved_docs": [
            {
                "source": "oracle_open_agent_spec.pdf",
                "title": "Oracle Open Agent Spec",
                "page": 7,
            }
        ],
    }

    rendered = format_response(payload)

    assert "Answer:" in rendered
    assert "1. [oracle_open_agent_spec.pdf]" in rendered
    assert "title=Oracle Open Agent Spec" in rendered
    assert "page=7" in rendered


def test_format_response_handles_missing_metadata_fields() -> None:
    """It should handle missing title and page metadata gracefully."""
    payload = {
        "output": "Answer text",
        "retrieved_docs": [{"source": "doc.pdf"}],
    }

    rendered = format_response(payload)

    assert "title=n/a" in rendered
    assert "page=n/a" in rendered
