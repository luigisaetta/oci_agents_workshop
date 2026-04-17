"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: CLI client for invoking the simple RAG FastAPI endpoint.
"""

from __future__ import annotations

import argparse
import json
from urllib import request


def main() -> None:
    """Read user request from CLI and call the RAG API."""
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

    with request.urlopen(http_request, timeout=120) as response:  # nosec B310
        body = response.read().decode("utf-8")

    print(body)


if __name__ == "__main__":
    main()
