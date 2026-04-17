"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: OCI model builders used by local agents.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_oci import ChatOCIGenAI, OCIGenAIEmbeddings


def build_llm(runtime_config: Dict[str, str]) -> ChatOCIGenAI:
    """Create a ChatOCIGenAI instance from runtime config."""
    return ChatOCIGenAI(
        model_id=runtime_config["OCI_MODEL_ID"],
        service_endpoint=runtime_config["OCI_SERVICE_ENDPOINT"],
        compartment_id=runtime_config["OCI_COMPARTMENT_ID"],
        provider=runtime_config["OCI_PROVIDER"],
        auth_type=runtime_config["OCI_AUTH_TYPE"],
        auth_profile=runtime_config["OCI_AUTH_PROFILE"],
        model_kwargs={"temperature": 0.0, "max_tokens": 4192},
    )


def build_embedding_client(runtime_config: Dict[str, str]) -> Any:
    """Create an OCIGenAIEmbeddings instance from runtime config."""
    return OCIGenAIEmbeddings(
        model_id=runtime_config["OCI_EMBED_MODEL_ID"],
        service_endpoint=runtime_config["OCI_SERVICE_ENDPOINT"],
        compartment_id=runtime_config["OCI_COMPARTMENT_ID"],
        auth_type=runtime_config["OCI_AUTH_TYPE"],
        auth_profile=runtime_config["OCI_AUTH_PROFILE"],
    )
