"""
Author: L. Saetta
Date last modified: 2026-04-20
License: MIT
Description: Minimal streaming Responses API example using OCI OpenAI-compatible client.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from dotenv import load_dotenv

from common.oci_openai_clients import build_oci_openai_client
from common.utils import collect_oci_runtime_config, print_oci_runtime_config

DEFAULT_MODEL_ID = "openai.gpt-oss-120B"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def collect_responses_runtime_config() -> Dict[str, str]:
    """Collect runtime settings required by this Responses API example.

    Returns:
        Dict[str, str]: Runtime config including OCI auth and OpenAI-compatible vars.

    Raises:
        ValueError: If required environment variables are missing.
    """
    runtime_config = collect_oci_runtime_config()

    base_url = os.getenv("OCI_OPENAI_BASE_URL", "").strip()
    if not base_url:
        raise ValueError("Set OCI_OPENAI_BASE_URL environment variable.")

    project_id = os.getenv("OCI_OPENAI_PROJECT_ID", "").strip()
    if not project_id:
        raise ValueError("Set OCI_OPENAI_PROJECT_ID environment variable.")

    runtime_config["OCI_OPENAI_BASE_URL"] = base_url
    runtime_config["OCI_OPENAI_PROJECT_ID"] = project_id
    return runtime_config


def stream_response_text(
    *,
    client: Any,
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> Iterator[str]:
    """Yield streamed text chunks from a Responses API call.

    Args:
        client: OpenAI-compatible client exposing ``responses.create``.
        prompt: Input text sent to the model.
        model_id: Model id to use for the request.

    Yields:
        str: Text chunks from streaming events.

    Raises:
        ValueError: If prompt is empty.
    """
    prompt_text = prompt.strip()
    if not prompt_text:
        raise ValueError("Prompt must not be empty.")

    stream = client.responses.create(
        model=model_id,
        input=prompt_text,
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        stream=True,
    )

    emitted_delta = False
    for event in stream:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            emitted_delta = True
            delta = getattr(event, "delta", "")
            if delta:
                yield str(delta)
        elif event_type == "response.output_text.done" and not emitted_delta:
            final_text = getattr(event, "text", "")
            if final_text:
                yield str(final_text)


def collect_streamed_output(stream: Iterable[str]) -> str:
    """Print streamed chunks and return the assembled output text.

    Args:
        stream: Iterable of output text chunks.

    Returns:
        str: Full concatenated output text.
    """
    text_parts: list[str] = []
    for chunk in stream:
        text_parts.append(chunk)
        print(chunk, end="", flush=True)
    return "".join(text_parts)


def main() -> None:
    """Read CLI input, run streaming Responses call, and print model output."""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Streaming Responses API example with OCI OpenAI-compatible client."
    )
    parser.add_argument(
        "request",
        type=str,
        help="Prompt text to send to the model.",
    )
    args = parser.parse_args()

    runtime_config = collect_responses_runtime_config()
    print_oci_runtime_config(runtime_config)
    print(f"  MODEL_ID={DEFAULT_MODEL_ID}")
    print("---")
    print("-------- Streaming Response --------")

    client = build_oci_openai_client(
        base_url=runtime_config["OCI_OPENAI_BASE_URL"],
        project_id=runtime_config["OCI_OPENAI_PROJECT_ID"],
        auth_profile=runtime_config["OCI_AUTH_PROFILE"],
    )

    stream = stream_response_text(client=client, prompt=args.request)
    collect_streamed_output(stream)
    print()


if __name__ == "__main__":
    main()
