"""
Author: L. Saetta
Date last modified: 2026-04-18
License: MIT
Description: Shared utility functions for OCI runtime config and stream output.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable


def collect_oci_runtime_config() -> Dict[str, str]:
    """Build reusable OCI runtime config from environment variables.

    Returns:
        Dict[str, str]: Runtime config used by model and embedding builders.

    Raises:
        ValueError: If ``OCI_COMPARTMENT_ID`` is missing.
    """
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
    """Print OCI runtime configuration values.

    Args:
        config: Runtime configuration dictionary to print.

    Returns:
        None: This function writes to stdout.
    """
    print("-------- OCI Runtime Configuration --------")
    for key, value in config.items():
        print(f"  {key}={value}")
    print("---")


def print_streamed_response(stream: Iterable[object]) -> None:
    """Print streamed model chunks as they arrive.

    Args:
        stream: Iterable yielding model chunks with optional ``content`` field.

    Returns:
        None: This function writes streaming output to stdout.
    """
    print("-------- Model Streaming Output --------")

    for chunk in stream:
        content = getattr(chunk, "content", "")
        if content:
            print(content, end="", flush=True)

    print("\n---")


def extract_text(response: Any) -> str:
    """Extract plain text from heterogeneous model response formats.

    Args:
        response: Model response object, string, or list-like structured payload.

    Returns:
        str: Extracted plain-text representation.
    """
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Some providers return segmented content with mixed item formats.
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        return "".join(text_parts)

    return str(content)


def sanitize_standalone_search_query(raw_text: str) -> str:
    """Extract a single standalone search query from verbose rewrite output.

    Args:
        raw_text: Raw model output potentially containing labels or explanations.

    Returns:
        str: Clean single-line search query suitable for vector-store retrieval.
    """
    cleaned = raw_text.strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    first_line = lines[0]
    lower_line = first_line.lower()
    prefix = "standalone search query:"
    if lower_line.startswith(prefix):
        first_line = first_line[len(prefix) :].strip()

    if first_line.startswith(("'", '"')) and first_line.endswith(("'", '"')):
        if len(first_line) >= 2:
            first_line = first_line[1:-1].strip()

    return first_line
