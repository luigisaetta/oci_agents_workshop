"""
Author: L. Saetta
Date last modified: 2026-04-20
License: MIT
Description: Unit tests for quickstart streaming Responses API example.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods

from types import SimpleNamespace
from typing import Any

import pytest

from quickstart import responses01


class _FakeResponses:
    """Fake responses namespace returning deterministic stream events."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        """Save call kwargs and return configured stream events."""
        self.last_kwargs = kwargs
        return self._events


class _FakeClient:
    """Fake OpenAI-compatible client exposing responses namespace."""

    def __init__(self, events: list[Any]) -> None:
        self.responses = _FakeResponses(events)


def test_collect_responses_runtime_config_requires_openai_vars(monkeypatch) -> None:
    """It should fail when OpenAI-compatible environment variables are missing."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.delenv("OCI_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OCI_OPENAI_PROJECT_ID", raising=False)

    with pytest.raises(ValueError, match="OCI_OPENAI_BASE_URL"):
        responses01.collect_responses_runtime_config()


def test_stream_response_text_streams_deltas_with_model_override() -> None:
    """It should use stream=True and pass through provided model id."""
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello "),
        SimpleNamespace(type="response.output_text.delta", delta="world"),
        SimpleNamespace(type="response.completed"),
    ]
    client = _FakeClient(events)

    chunks = list(
        responses01.stream_response_text(
            client=client,
            prompt="Say hello",
            compartment_id="ocid1.compartment.oc1..example",
            model_id="openai.gpt-oss-120B",
        )
    )

    assert "".join(chunks) == "Hello world"
    assert client.responses.last_kwargs["model"] == "openai.gpt-oss-120B"
    assert client.responses.last_kwargs["extra_headers"] == {
        "opc-compartment-id": "ocid1.compartment.oc1..example"
    }
    assert client.responses.last_kwargs["stream"] is True


def test_stream_response_text_uses_done_when_no_delta() -> None:
    """It should use output_text.done text when delta events are not present."""
    events = [SimpleNamespace(type="response.output_text.done", text="Final answer")]
    client = _FakeClient(events)

    chunks = list(
        responses01.stream_response_text(
            client=client,
            prompt="Answer",
            compartment_id="ocid1.compartment.oc1..example",
            model_id="meta.llama-3.3-70b-instruct",
        )
    )

    assert chunks == ["Final answer"]


def test_main_reads_prompt_from_cli_and_prints_stream(monkeypatch, capsys) -> None:
    """It should read CLI input and print streamed response content."""
    monkeypatch.setattr(
        responses01,
        "collect_responses_runtime_config",
        lambda: {
            "OCI_AUTH_PROFILE": "DEFAULT",
            "OCI_OPENAI_BASE_URL": "https://example.test/v1",
            "OCI_OPENAI_PROJECT_ID": "proj_123",
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
            "OCI_MODEL_ID": "openai.gpt-oss-120B",
        },
    )
    monkeypatch.setattr(responses01, "print_oci_runtime_config", lambda _cfg: None)
    monkeypatch.setattr(responses01, "build_oci_openai_client", lambda **_kw: object())
    monkeypatch.setattr(
        responses01,
        "stream_response_text",
        lambda **kwargs: iter(
            [
                (
                    f"{kwargs['model_id']}|{kwargs['compartment_id']}"
                    f"|Echo: {kwargs['prompt']}"
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["quickstart/responses01.py", "prompt from cli"],
    )

    responses01.main()

    printed = capsys.readouterr().out
    assert "Streaming Response" in printed
    assert (
        "openai.gpt-oss-120B|ocid1.compartment.oc1..example|Echo: prompt from cli"
        in printed
    )
