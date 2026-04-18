"""
Author: L. Saetta
Date last modified: 2026-04-18
License: MIT
Description: Unit tests for the simple three-step LangGraph agent.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods

import json
from types import SimpleNamespace

import pytest

from quickstart import agent01
from common.utils import extract_text


class _FakeLlm:
    """Minimal fake model used by tests."""

    def __init__(self, response: object) -> None:
        self._response = response

    def invoke(self, prompt: str) -> object:
        """Return a predictable fake response."""
        if not prompt:
            raise ValueError("Prompt must not be empty in test.")
        return self._response


def _build_fake_llm(_: dict[str, str]) -> _FakeLlm:
    """Default fake builder returning a fixed string response."""
    return _FakeLlm("Hello from fake model")


def test_extract_text_from_string() -> None:
    """It should return the same text when response content is already a string."""
    assert extract_text("Hello") == "Hello"


def test_extract_text_from_object_with_list_content() -> None:
    """It should merge text parts from list content."""
    response = SimpleNamespace(content=[{"text": "Hello "}, {"text": "Rome"}])
    assert extract_text(response) == "Hello Rome"


def test_run_agent_returns_json_dictionary(monkeypatch) -> None:
    """It should run the full graph and return output as dict."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    result = agent01.run_agent("Tell me about Rome", llm_builder=_build_fake_llm)
    assert result == {"output": "Hello from fake model"}


def test_run_agent_logs_all_step_names(monkeypatch, caplog) -> None:
    """It should emit log messages for all graph steps."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    caplog.set_level("INFO")

    agent01.run_agent("Tell me about Rome", llm_builder=_build_fake_llm)

    messages = [record.message for record in caplog.records]
    assert "Running step: step1_log_input" in messages
    assert "Running step: step2_invoke_model" in messages
    assert "Running step: step3_build_json_output" in messages


def test_step2_raises_when_compartment_is_missing(monkeypatch) -> None:
    """It should fail fast when OCI_COMPARTMENT_ID is not set."""
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    step = agent01.Step2InvokeModel(llm_builder=_build_fake_llm)
    with pytest.raises(ValueError, match="OCI_COMPARTMENT_ID"):
        step.invoke({"user_input": "Test"})


def test_main_prints_json_output(monkeypatch, capsys) -> None:
    """It should print the final JSON payload to stdout."""
    monkeypatch.setattr(agent01, "run_agent", lambda _: {"output": "ok"})
    monkeypatch.setattr("sys.argv", ["quickstart/agent01.py", "hello"])
    agent01.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip()) == {"output": "ok"}
