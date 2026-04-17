"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Minimal OCI embedding example for a list of texts using langchain-oci.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

from oci_models import build_embedding_client
from utils import collect_oci_runtime_config, print_oci_runtime_config


def build_embedding_runtime_config() -> Dict[str, str]:
    """Build OCI runtime config and require a dedicated embedding model id."""
    runtime_config = collect_oci_runtime_config()
    embed_model_id = os.getenv("OCI_EMBED_MODEL_ID", "").strip()
    if not embed_model_id:
        raise ValueError("Set OCI_EMBED_MODEL_ID environment variable.")

    runtime_config["OCI_EMBED_MODEL_ID"] = embed_model_id
    return runtime_config


def generate_embeddings(
    texts: Sequence[str],
    embedding_client: Any,
) -> List[List[float]]:
    """Generate embeddings for the provided text list."""
    text_list = list(texts)
    if not text_list:
        raise ValueError("Provide at least one text.")
    return embedding_client.embed_documents(text_list)


def summarize_embeddings(
    texts: Sequence[str], vectors: Sequence[Sequence[float]]
) -> List[Dict[str, Any]]:
    """Build a compact summary of generated embeddings."""
    if len(texts) != len(vectors):
        raise ValueError("Texts and vectors must have the same length.")

    summary: List[Dict[str, Any]] = []
    for index, (text, vector) in enumerate(zip(texts, vectors, strict=True), start=1):
        summary.append(
            {
                "index": index,
                "text": text,
                "dimension": len(vector),
                "preview": list(vector[:5]),
            }
        )
    return summary


def main() -> None:
    """Run a minimal embedding generation example against OCI Generative AI."""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    runtime_config = build_embedding_runtime_config()
    print_oci_runtime_config(runtime_config)

    texts = [
        "Rome is the capital of Italy.",
        "LangChain can integrate with OCI Enterprise AI.",
        "Dedicated AI Clusters can expose inference endpoints.",
    ]

    embedding_client = build_embedding_client(runtime_config)
    vectors = generate_embeddings(texts, embedding_client=embedding_client)
    summary = summarize_embeddings(texts, vectors)

    print("-------- Embedding Output Summary --------")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
