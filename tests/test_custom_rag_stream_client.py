"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Unit tests for custom RAG streaming CLI event parsing/rendering.
"""

from __future__ import annotations

import pytest

from custom_rag_agent.stream_client import parse_sse_data_line, render_stream_event


def test_parse_sse_data_line_parses_json_payload() -> None:
    """It should parse JSON content from SSE data frames."""
    event = parse_sse_data_line('data: {"event":"completed","output":"ok"}')

    assert event == {"event": "completed", "output": "ok"}


def test_parse_sse_data_line_ignores_non_data_lines() -> None:
    """It should return None for non-data SSE lines."""
    assert parse_sse_data_line("event: message") is None


def test_parse_sse_data_line_raises_on_invalid_json() -> None:
    """It should raise ValueError when data payload is not valid JSON."""
    with pytest.raises(ValueError):
        parse_sse_data_line("data: not-json")


def test_render_stream_event_for_retrieval_results() -> None:
    """It should include count and first retrieved metadata in output."""
    text = render_stream_event(
        {
            "event": "retrieval_results",
            "retrieved_docs": [{"source": "doc-1"}, {"source": "doc-2"}],
        }
    )

    assert text == '[retrieval_results] 2 documents, first={"source": "doc-1"}'
