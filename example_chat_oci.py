"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Minimal OCI chat streaming example using LangChain ChatOCIGenAI.
"""

from pathlib import Path

from dotenv import load_dotenv

from langchain_oci import ChatOCIGenAI

from utils import (
    collect_oci_runtime_config,
    print_oci_runtime_config,
    print_streamed_response,
)


def main() -> None:
    """Run a streaming chat completion against OCI Generative AI."""
    # Load environment variables from .env in the same directory as this script.
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    # Collect all OCI runtime settings in one place for reuse and logging.
    runtime_config = collect_oci_runtime_config()

    model_id = runtime_config["OCI_MODEL_ID"]
    service_endpoint = runtime_config["OCI_SERVICE_ENDPOINT"]
    compartment_id = runtime_config["OCI_COMPARTMENT_ID"]
    provider = runtime_config["OCI_PROVIDER"]
    auth_type = runtime_config["OCI_AUTH_TYPE"]
    auth_profile = runtime_config["OCI_AUTH_PROFILE"]

    # Build model client arguments from runtime configuration.
    llm_kwargs = {
        "model_id": model_id,
        "service_endpoint": service_endpoint,
        "compartment_id": compartment_id,
        "provider": provider,
        "auth_type": auth_type,
        "auth_profile": auth_profile,
        "model_kwargs": {"temperature": 0.0, "max_tokens": 4192},
    }

    llm = ChatOCIGenAI(**llm_kwargs)

    # Show effective runtime configuration before model output.
    print_oci_runtime_config(runtime_config)

    # invoke the model in streaming mode
    stream = llm.stream("Tell me something about Rome, Italy.")

    # print the streaming response as it arrives
    print_streamed_response(stream)


if __name__ == "__main__":
    main()
