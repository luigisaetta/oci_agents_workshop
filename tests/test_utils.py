"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Unit tests for utility helpers used by the OCI chat example.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.utils import (
    collect_oci_runtime_config,
    print_oci_runtime_config,
    print_streamed_response,
)


def test_collect_oci_runtime_config_raises_when_compartment_is_missing(
    monkeypatch,
) -> None:
    """It should raise when OCI_COMPARTMENT_ID is not set."""
    monkeypatch.delenv("OCI_MODEL_ID", raising=False)
    monkeypatch.delenv("OCI_REGION", raising=False)
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    monkeypatch.delenv("OCI_PROVIDER", raising=False)
    monkeypatch.delenv("OCI_AUTH_TYPE", raising=False)
    monkeypatch.delenv("OCI_AUTH_PROFILE", raising=False)

    with pytest.raises(ValueError, match="OCI_COMPARTMENT_ID"):
        collect_oci_runtime_config()


def test_collect_oci_runtime_config_with_defaults(monkeypatch) -> None:
    """It should return expected default values when optional env vars are not set."""
    monkeypatch.delenv("OCI_MODEL_ID", raising=False)
    monkeypatch.delenv("OCI_REGION", raising=False)
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.delenv("OCI_PROVIDER", raising=False)
    monkeypatch.delenv("OCI_AUTH_TYPE", raising=False)
    monkeypatch.delenv("OCI_AUTH_PROFILE", raising=False)

    config = collect_oci_runtime_config()

    assert config["OCI_MODEL_ID"] == "meta.llama-3.3-70b-instruct"
    assert config["OCI_REGION"] == "us-chicago-1"
    assert (
        config["OCI_SERVICE_ENDPOINT"]
        == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    )
    assert config["OCI_COMPARTMENT_ID"] == "ocid1.compartment.oc1..example"
    assert config["OCI_PROVIDER"] == "generic"
    assert config["OCI_AUTH_TYPE"] == "API_KEY"
    assert config["OCI_AUTH_PROFILE"] == "DEFAULT"


def test_collect_oci_runtime_config_with_env_overrides(monkeypatch) -> None:
    """It should use environment overrides when they are provided."""
    monkeypatch.setenv("OCI_MODEL_ID", "custom-model")
    monkeypatch.setenv("OCI_REGION", "eu-frankfurt-1")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_PROVIDER", "oci")
    monkeypatch.setenv("OCI_AUTH_TYPE", " API_KEY ")
    monkeypatch.setenv("OCI_AUTH_PROFILE", "MYPROFILE")

    config = collect_oci_runtime_config()

    assert config["OCI_MODEL_ID"] == "custom-model"
    assert config["OCI_REGION"] == "eu-frankfurt-1"
    assert (
        config["OCI_SERVICE_ENDPOINT"]
        == "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
    )
    assert config["OCI_COMPARTMENT_ID"] == "ocid1.compartment.oc1..example"
    assert config["OCI_PROVIDER"] == "oci"
    assert config["OCI_AUTH_TYPE"] == "API_KEY"
    assert config["OCI_AUTH_PROFILE"] == "MYPROFILE"


def test_print_oci_runtime_config_outputs_expected_lines(capsys) -> None:
    """It should print all keys and values in the provided runtime config."""
    config = {"OCI_MODEL_ID": "m1", "OCI_REGION": "r1"}

    print_oci_runtime_config(config)
    captured = capsys.readouterr()

    assert "-------- OCI Runtime Configuration --------" in captured.out
    assert "  OCI_MODEL_ID=m1" in captured.out
    assert "  OCI_REGION=r1" in captured.out
    assert "---" in captured.out


def test_print_streamed_response_prints_only_non_empty_chunks(capsys) -> None:
    """It should print only chunks containing content."""
    stream = [
        SimpleNamespace(content="Hello"),
        SimpleNamespace(content=" "),
        SimpleNamespace(content="Rome"),
        SimpleNamespace(content=""),
        object(),
    ]

    print_streamed_response(stream)
    captured = capsys.readouterr()

    assert "-------- Model Streaming Output --------" in captured.out
    assert "Hello Rome" in captured.out
    assert captured.out.rstrip().endswith("---")
