"""
Author: L. Saetta
Date last modified: 2026-04-18
License: MIT
Description: Unit tests for the embedding01 OCI embedding example.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods

import pytest

from quickstart import embedding01


class _FakeEmbeddingClient:
    """Minimal fake embedding client used by tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one 3-dim vector for each input text."""
        return [[float(index), 0.5, 1.5] for index, _ in enumerate(texts, start=1)]


def test_build_embedding_runtime_config_requires_embed_model_id(monkeypatch) -> None:
    """It should fail fast when OCI_EMBED_MODEL_ID is not configured."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.delenv("OCI_EMBED_MODEL_ID", raising=False)

    with pytest.raises(ValueError, match="OCI_EMBED_MODEL_ID"):
        embedding01.build_embedding_runtime_config()


def test_build_embedding_runtime_config_with_embed_model_id(monkeypatch) -> None:
    """It should include OCI_EMBED_MODEL_ID in runtime config."""
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.setenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-v3.0")

    config = embedding01.build_embedding_runtime_config()

    assert config["OCI_COMPARTMENT_ID"] == "ocid1.compartment.oc1..example"
    assert config["OCI_EMBED_MODEL_ID"] == "cohere.embed-english-v3.0"


def test_generate_embeddings_returns_vectors() -> None:
    """It should return one vector per input text."""
    fake_client = _FakeEmbeddingClient()
    vectors = embedding01.generate_embeddings(["a", "b"], embedding_client=fake_client)

    assert len(vectors) == 2
    assert vectors[0] == [1.0, 0.5, 1.5]


def test_generate_embeddings_raises_on_empty_input() -> None:
    """It should raise when no input texts are provided."""
    fake_client = _FakeEmbeddingClient()

    with pytest.raises(ValueError, match="at least one text"):
        embedding01.generate_embeddings([], embedding_client=fake_client)


def test_summarize_embeddings_builds_expected_output() -> None:
    """It should expose dimension and short preview for each vector."""
    texts = ["first", "second"]
    vectors = [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0, 4.0]]

    summary = embedding01.summarize_embeddings(texts, vectors)

    assert summary[0]["index"] == 1
    assert summary[0]["dimension"] == 3
    assert summary[0]["preview"] == [0.1, 0.2, 0.3]
    assert summary[1]["index"] == 2
    assert summary[1]["dimension"] == 4


def test_summarize_embeddings_raises_on_length_mismatch() -> None:
    """It should raise when texts and vectors lengths differ."""
    with pytest.raises(ValueError, match="same length"):
        embedding01.summarize_embeddings(["only-one"], [[1.0], [2.0]])
