"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: FastAPI server exposing the simple RAG agent over HTTP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from simple_rag_agent.rag_agent import run_rag_agent

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

app = FastAPI(title="Simple RAG Agent API")


class InvokeRequest(BaseModel):
    """Input payload for RAG invocation."""

    request: str


class InvokeResponse(BaseModel):
    """Output payload for RAG invocation."""

    output: str
    retrieved_docs: List[dict[str, str]]


@app.post("/invoke", response_model=InvokeResponse)
def invoke_agent(payload: InvokeRequest) -> InvokeResponse:
    """Invoke the simple RAG agent and return its JSON output."""
    result = run_rag_agent(payload.request)
    return InvokeResponse(
        output=result["output"],
        retrieved_docs=result["retrieved_docs"],
    )
