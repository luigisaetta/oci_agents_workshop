"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: CLI client for streaming custom RAG events from FastAPI SSE endpoint.
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib import request


def parse_sse_data_line(line: str) -> dict[str, Any] | None:
    """Parse an SSE data line into a JSON payload dictionary.

    Args:
        line: Raw SSE line, potentially prefixed by ``data:``.

    Returns:
        dict[str, Any] | None: Parsed JSON payload or ``None`` when line is
            not a data frame.

    Raises:
        ValueError: If the data payload is not valid JSON.
    """
    if not line.startswith("data:"):
        return None

    payload_raw = line[len("data:") :].strip()
    if not payload_raw:
        return None
    return json.loads(payload_raw)


def render_stream_event(event_payload: dict[str, Any]) -> str:
    """Render a non-token stream event as human-readable text.

    Args:
        event_payload: Parsed event dictionary.

    Returns:
        str: Readable line for terminal output.
    """
    event_type = str(event_payload.get("event", "unknown"))

    if event_type in {"step_started", "step_completed"}:
        step = str(event_payload.get("step", "unknown"))
        return f"[{event_type}] {step}"

    if event_type == "retrieval_results":
        docs = event_payload.get("retrieved_docs", [])
        if not docs:
            return "[retrieval_results] 0 documents"
        first_doc = docs[0]
        return (
            f"[retrieval_results] {len(docs)} documents, "
            f"first={json.dumps(first_doc, ensure_ascii=False)}"
        )

    if event_type == "completed":
        return "\n[completed] response fully generated"

    return f"[{event_type}]"


def main() -> None:
    """Parse CLI arguments, call streaming endpoint, and print live output.

    Returns:
        None: This function writes streaming output to stdout.
    """
    parser = argparse.ArgumentParser(description="Custom RAG streaming API client.")
    parser.add_argument("request", type=str, help="Prompt text to send to the API.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/invoke/stream",
        help="Streaming RAG API URL.",
    )
    args = parser.parse_args()

    payload = json.dumps({"request": args.request, "history": []}).encode("utf-8")
    http_request = request.Request(
        args.url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    print("=== Custom RAG Streaming Response ===")
    print("\nAnswer:\n")

    with request.urlopen(http_request, timeout=240) as response:  # nosec B310
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            event_payload = parse_sse_data_line(line)
            if event_payload is None:
                continue

            if event_payload.get("event") == "final_answer_token":
                token = str(event_payload.get("token", ""))
                if token:
                    print(token, end="", flush=True)
                continue

            message = render_stream_event(event_payload)
            print(message)

    print("\n---")


if __name__ == "__main__":
    main()
