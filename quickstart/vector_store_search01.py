"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Semantic vector store search example using OpenAI SDK on OCI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document

from common.oci_openai_clients import build_oci_openai_client
from common.utils import collect_oci_runtime_config, print_oci_runtime_config


def collect_vector_store_runtime_config() -> Dict[str, str]:
    """Collect runtime settings required by this vector store example.

    Returns:
        Dict[str, str]: Runtime config with OCI and vector store settings.

    Raises:
        ValueError: If required environment variables are missing.
    """
    runtime_config = collect_oci_runtime_config()
    base_url = os.getenv("OCI_OPENAI_BASE_URL", "").strip()
    if not base_url:
        raise ValueError("Set OCI_OPENAI_BASE_URL environment variable.")

    vector_store_id = os.getenv("OCI_VECTOR_STORE_ID", "").strip()
    if not vector_store_id:
        raise ValueError("Set OCI_VECTOR_STORE_ID environment variable.")

    project_id = os.getenv("OCI_OPENAI_PROJECT_ID", "").strip()
    if not project_id:
        raise ValueError("Set OCI_OPENAI_PROJECT_ID environment variable.")

    runtime_config["OCI_OPENAI_BASE_URL"] = base_url
    runtime_config["OCI_VECTOR_STORE_ID"] = vector_store_id
    runtime_config["OCI_OPENAI_PROJECT_ID"] = project_id
    return runtime_config


def _extract_result_text(result_item: Any) -> str:
    """Extract plain text from a vector store search result item.

    Args:
        result_item: Single item from ``vector_stores.search(...).data``.

    Returns:
        str: Extracted chunk text.
    """
    content = getattr(result_item, "content", None)
    if isinstance(content, list):
        text_parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                text_parts.append(chunk)
                continue
            text = getattr(chunk, "text", None)
            if text is not None:
                text_parts.append(str(text))
                continue
            if isinstance(chunk, dict):
                chunk_text = chunk.get("text")
                if chunk_text is not None:
                    text_parts.append(str(chunk_text))
        if text_parts:
            return "\n".join(text_parts)
    return ""


def _build_document_from_result(result_item: Any) -> Document:
    """Convert a vector store result item to a LangChain ``Document``.

    Args:
        result_item: Single item from ``vector_stores.search(...).data``.

    Returns:
        Document: Document with extracted chunk text and normalized metadata.
    """
    metadata: Dict[str, Any] = {}
    additional_properties = getattr(result_item, "additional_properties", {}) or {}
    if not isinstance(additional_properties, dict):
        additional_properties = {}

    pages = additional_properties.get("page_numbers")
    if isinstance(pages, list) and pages:
        page_value: int | str | None = pages[0]
    else:
        page_value = additional_properties.get("page")

    filename = getattr(result_item, "filename", None)
    metadata["source"] = additional_properties.get("source") or filename or "unknown"
    metadata["title"] = additional_properties.get("title") or filename or "n/a"
    metadata["page"] = page_value if page_value is not None else "n/a"
    metadata["score"] = float(getattr(result_item, "score", 0.0) or 0.0)

    file_id = getattr(result_item, "file_id", None)
    if file_id:
        metadata["file_id"] = file_id

    chunk_id = additional_properties.get("chunk_id")
    if chunk_id:
        metadata["chunk_id"] = chunk_id

    return Document(page_content=_extract_result_text(result_item), metadata=metadata)


def semantic_search_vector_store(
    *,
    query: str,
    vector_store_id: str,
    project_id: str,
    client: Any,
    max_num_results: int = 4,
) -> List[Document]:
    """Run semantic search over an OCI vector store and return LangChain docs.

    Args:
        query: Natural-language retrieval query.
        vector_store_id: OCI vector store identifier.
        project_id: OCI project identifier used by OpenAI-compatible APIs.
        client: OpenAI-compatible client exposing ``vector_stores.search``.
        max_num_results: Maximum number of results requested.

    Returns:
        List[Document]: Retrieved results converted to LangChain documents.

    Raises:
        ValueError: If query is empty or max_num_results is not positive.
    """
    query_text = query.strip()
    if not query_text:
        raise ValueError("Query must not be empty.")
    if max_num_results < 1:
        raise ValueError("max_num_results must be a positive integer.")

    search_results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query_text,
        max_num_results=max_num_results,
        extra_headers={"OpenAI-Project": project_id},
    )

    sorted_items = sorted(
        getattr(search_results, "data", []),
        key=lambda item: getattr(item, "score", 0.0) or 0.0,
        reverse=True,
    )
    return [_build_document_from_result(item) for item in sorted_items]


def build_retrieved_docs_metadata(documents: List[Document]) -> List[Dict[str, Any]]:
    """Build a JSON-serializable metadata list from LangChain documents.

    Args:
        documents: Retrieved LangChain documents.

    Returns:
        List[Dict[str, Any]]: Metadata list compatible with custom_rag_agent output.
    """
    return [dict(document.metadata) for document in documents]


def main() -> None:
    """Run semantic vector store search and print metadata output as JSON."""
    parser = argparse.ArgumentParser(
        description="Run semantic search against an OCI vector store."
    )
    parser.add_argument(
        "query",
        help="Natural-language search query for the vector store.",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

    runtime_config = collect_vector_store_runtime_config()
    print_oci_runtime_config(runtime_config)

    client = build_oci_openai_client(
        base_url=runtime_config["OCI_OPENAI_BASE_URL"],
        project_id=runtime_config["OCI_OPENAI_PROJECT_ID"],
        auth_profile=runtime_config["OCI_AUTH_PROFILE"],
    )
    documents = semantic_search_vector_store(
        query=args.query,
        vector_store_id=runtime_config["OCI_VECTOR_STORE_ID"],
        project_id=runtime_config["OCI_OPENAI_PROJECT_ID"],
        client=client,
        max_num_results=int(os.getenv("SIMPLE_RAG_TOP_K", "4")),
    )
    retrieved_docs = build_retrieved_docs_metadata(documents)

    print("-------- Retrieved Docs Metadata --------")
    print(json.dumps(retrieved_docs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
