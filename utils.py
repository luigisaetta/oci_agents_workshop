"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Shared utility functions for OCI runtime config and stream output.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable


def collect_oci_runtime_config() -> Dict[str, str]:
    """Build a reusable OCI runtime config dictionary from environment variables."""
    model_id = os.getenv("OCI_MODEL_ID", "meta.llama-3.3-70b-instruct")
    region = os.getenv("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    compartment_id = os.getenv("OCI_COMPARTMENT_ID", "")
    if not compartment_id:
        raise ValueError("Set OCI_COMPARTMENT_ID environment variable.")
    provider = os.getenv("OCI_PROVIDER", "generic")
    auth_type = os.getenv("OCI_AUTH_TYPE", "API_KEY").strip()
    auth_profile = os.getenv("OCI_AUTH_PROFILE", "DEFAULT")

    return {
        "OCI_MODEL_ID": model_id,
        "OCI_REGION": region,
        "OCI_SERVICE_ENDPOINT": service_endpoint,
        "OCI_COMPARTMENT_ID": compartment_id,
        "OCI_PROVIDER": provider,
        "OCI_AUTH_TYPE": auth_type,
        "OCI_AUTH_PROFILE": auth_profile,
    }


def print_oci_runtime_config(config: Dict[str, str]) -> None:
    """Print the OCI runtime config used for model execution."""
    print("-------- OCI Runtime Configuration --------")
    for key, value in config.items():
        print(f"  {key}={value}")
    print("---")


def print_streamed_response(stream: Iterable[object]) -> None:
    """Print streamed model chunks as they arrive."""
    print("-------- Model Streaming Output --------")

    for chunk in stream:
        content = getattr(chunk, "content", "")
        if content:
            print(content, end="", flush=True)

    print("\n---")


def extract_text(response: Any) -> str:
    """Extract plain text from a model response object."""
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        return "".join(text_parts)

    return str(content)
