"""
Author: L. Saetta
Date last modified: 2026-04-18
License: MIT
Description: CLI client for invoking the simple RAG FastAPI endpoint.
"""

from __future__ import annotations

import argparse
import json
from urllib import request


def format_response(response_payload: dict) -> str:
    """Format API JSON payload into a readable CLI string.

    Args:
        response_payload: Parsed JSON object returned by the API.

    Returns:
        str: Multi-line text ready to print on terminal.
    """
    output_text = str(response_payload.get("output", "")).strip()
    retrieved_docs = response_payload.get("retrieved_docs", [])

    lines = [
        "=== Simple RAG Response ===",
        "",
        "Answer:",
        output_text or "(empty)",
        "",
        "Retrieved Documents:",
    ]

    if not retrieved_docs:
        lines.append("- none")
        return "\n".join(lines)

    for index, item in enumerate(retrieved_docs, start=1):
        source = str(item.get("source", "unknown"))
        title = str(item.get("title", "n/a"))
        page = item.get("page", "n/a")
        lines.append(f"{index}. [{source}] title={title}, page={page}")

    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, call the API endpoint, and print formatted output.

    Returns:
        None: This function prints the final result to stdout.
    """
    parser = argparse.ArgumentParser(description="Simple RAG API client.")
    parser.add_argument("request", type=str, help="Prompt text to send to the API.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/invoke",
        help="RAG API URL.",
    )
    args = parser.parse_args()

    payload = json.dumps({"request": args.request}).encode("utf-8")
    http_request = request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Keep a generous timeout to support slower model responses.
    with request.urlopen(http_request, timeout=120) as response:  # nosec B310
        body = response.read().decode("utf-8")

    response_payload = json.loads(body)
    print(format_response(response_payload))


if __name__ == "__main__":
    main()
