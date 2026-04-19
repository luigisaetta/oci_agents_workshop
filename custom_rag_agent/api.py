"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: FastAPI server exposing the custom RAG agent over HTTP.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, List

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from custom_rag_agent.pdf_loader import build_pdf_vector_store, list_pdf_files
from custom_rag_agent.rag_agent import (
    build_initialized_vector_store,
    run_rag_agent,
    stream_rag_agent_events,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Load and index vector store once when the API starts.

    Args:
        app_instance: FastAPI application instance.

    Yields:
        None: Control is handed back to FastAPI for app serving.
    """
    input_dir = Path(__file__).resolve().parent.parent / "input_pdf"
    pdf_files = list_pdf_files(input_dir)

    # Prefer PDF-backed retrieval when local source documents are available.
    if pdf_files:
        app_instance.state.vector_store = build_pdf_vector_store(input_dir=input_dir)
    else:
        app_instance.state.vector_store = build_initialized_vector_store()

    yield


app = FastAPI(title="Custom RAG Agent API", lifespan=lifespan)

# Allow local web clients (for example Next.js on :3000) by default.
cors_origins_raw = os.getenv(
    "SIMPLE_RAG_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
cors_origins = [
    origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvokeRequest(BaseModel):
    """Request payload for RAG invocation.

    Attributes:
        request: User question sent to the RAG pipeline.
        history: Optional previous conversation messages.
        top_k: Optional request-level retrieval top_k override.
    """

    request: str
    history: List[dict[str, str]] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1)


class InvokeResponse(BaseModel):
    """Response payload returned by the RAG endpoint.

    Attributes:
        output: Final generated answer.
        retrieved_docs: Metadata list for retrieved context documents.
    """

    output: str
    retrieved_docs: List[dict[str, Any]]


def _as_sse_data(payload: dict[str, Any]) -> str:
    """Convert an event payload into a Server-Sent Events data frame.

    Args:
        payload: Event payload dictionary.

    Returns:
        str: Encoded SSE chunk ending with a blank line.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_sse_events(
    payload: InvokeRequest,
    vector_store: Any,
) -> Iterator[str]:
    """Yield SSE-encoded RAG events for a streaming invocation.

    Args:
        payload: Request payload.
        vector_store: Shared vector store from app startup.

    Yields:
        str: SSE-formatted data messages.
    """
    for event in stream_rag_agent_events(
        payload.request,
        history=payload.history,
        vector_store=vector_store,
        top_k=payload.top_k,
    ):
        yield _as_sse_data(event)


@app.post("/invoke", response_model=InvokeResponse)
def invoke_agent(payload: InvokeRequest, request: Request) -> InvokeResponse:
    """Invoke the RAG agent and return structured JSON response.

    Args:
        payload: Request body containing the user prompt.
        request: FastAPI request object used to access app state.

    Returns:
        InvokeResponse: Generated answer and retrieved docs metadata.
    """
    result = run_rag_agent(
        payload.request,
        history=payload.history,
        vector_store=request.app.state.vector_store,
        top_k=payload.top_k,
    )
    return InvokeResponse(
        output=result["output"],
        retrieved_docs=result["retrieved_docs"],
    )


@app.post("/invoke/stream")
def invoke_agent_stream(payload: InvokeRequest, request: Request) -> StreamingResponse:
    """Invoke the agent and return step-by-step Server-Sent Events.

    Args:
        payload: Request body containing user prompt and history.
        request: FastAPI request object used to access app state.

    Returns:
        StreamingResponse: SSE response with progress and token events.
    """
    stream = _stream_sse_events(payload, request.app.state.vector_store)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
