"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Minimal two-step LangGraph RAG agent using OCI embeddings and LLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable
from langgraph.graph import END, StateGraph

from oci_models import build_embedding_client, build_llm
from simple_rag_agent.prompts import build_answer_prompt
from utils import collect_oci_runtime_config, extract_text

DEFAULT_DOCUMENTS: Sequence[Document] = (
    Document(
        page_content=(
            "OCI Generative AI provides foundation models and inference endpoints."
        ),
        metadata={"source": "doc-1"},
    ),
    Document(
        page_content=(
            "Dedicated AI Clusters can be used for enterprise-grade model serving in OCI."
        ),
        metadata={"source": "doc-2"},
    ),
    Document(
        page_content=(
            "LangChain and LangGraph can orchestrate multi-step AI workflows."
        ),
        metadata={"source": "doc-3"},
    ),
    Document(
        page_content=(
            "OCI Identity and Access Management controls access with groups and policies."
        ),
        metadata={"source": "doc-4"},
    ),
    Document(
        page_content=(
            "OCI Object Storage can be used to store unstructured data for AI workloads."
        ),
        metadata={"source": "doc-5"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation combines retrieval with model generation."
        ),
        metadata={"source": "doc-6"},
    ),
    Document(
        page_content=(
            "Embeddings map text into vectors that can be compared by similarity."
        ),
        metadata={"source": "doc-7"},
    ),
    Document(
        page_content=("FastAPI can expose AI workflows through simple HTTP endpoints."),
        metadata={"source": "doc-8"},
    ),
    Document(
        page_content=(
            "Uvicorn is a common ASGI server used to run FastAPI applications."
        ),
        metadata={"source": "doc-9"},
    ),
    Document(
        page_content=(
            "A minimal RAG pipeline can be built with semantic search and answer generation."
        ),
        metadata={"source": "doc-10"},
    ),
)

TOP_K_RESULTS = 4


class RagState(TypedDict, total=False):
    """State shared by all RAG graph steps."""

    user_input: str
    runtime_config: Dict[str, str]
    documents: List[Document]
    output: str


def _collect_rag_runtime_config() -> Dict[str, str]:
    """Collect runtime config and require OCI_EMBED_MODEL_ID."""
    runtime_config = collect_oci_runtime_config()
    embed_model_id = os.getenv("OCI_EMBED_MODEL_ID", "").strip()
    if not embed_model_id:
        raise ValueError("Set OCI_EMBED_MODEL_ID environment variable.")
    runtime_config["OCI_EMBED_MODEL_ID"] = embed_model_id
    return runtime_config


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    numerator = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    norm_a = sum(value * value for value in vector_a) ** 0.5
    norm_b = sum(value * value for value in vector_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


# pylint: disable=too-few-public-methods
class SemanticSearcher(RunnableSerializable[RagState, RagState]):
    """Step 1: retrieve most relevant documents with semantic similarity."""

    def __init__(
        self,
        base_documents: Sequence[Document] = DEFAULT_DOCUMENTS,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        self._base_documents = list(base_documents)
        self._top_k = top_k

    def invoke(self, state: RagState, _config: Any = None, **_kwargs: Any) -> RagState:
        """Retrieve top relevant documents for the user request."""
        logging.info("START SemanticSearcher")

        runtime_config = _collect_rag_runtime_config()
        embedding_client = build_embedding_client(runtime_config)

        query_vector = embedding_client.embed_query(state["user_input"])
        doc_vectors = embedding_client.embed_documents(
            [document.page_content for document in self._base_documents]
        )

        ranked_pairs = []
        for document, vector in zip(self._base_documents, doc_vectors, strict=True):
            score = _cosine_similarity(query_vector, vector)
            ranked_pairs.append((score, document))

        ranked_pairs.sort(key=lambda item: item[0], reverse=True)
        top_documents = [document for _, document in ranked_pairs[: self._top_k]]

        updated_state: RagState = dict(state)
        updated_state["runtime_config"] = runtime_config
        updated_state["documents"] = top_documents

        logging.info("END SemanticSearcher")
        return updated_state


# pylint: disable=too-few-public-methods
class AnswerGenerator(RunnableSerializable[RagState, RagState]):
    """Step 2: generate final answer from retrieved documents."""

    def invoke(self, state: RagState, _config: Any = None, **_kwargs: Any) -> RagState:
        """Build prompt from context and invoke LLM."""
        logging.info("START AnswerGenerator")

        runtime_config = state.get("runtime_config", _collect_rag_runtime_config())
        llm = build_llm(runtime_config)

        context = "\n\n".join(document.page_content for document in state["documents"])
        prompt = build_answer_prompt(user_input=state["user_input"], context=context)
        response = llm.invoke(prompt)

        updated_state: RagState = dict(state)
        updated_state["output"] = extract_text(response)

        logging.info("END AnswerGenerator")
        return updated_state


def build_rag_graph():
    """Build and compile the simple two-step RAG graph."""
    graph_builder = StateGraph(RagState)
    graph_builder.add_node("semantic_searcher", SemanticSearcher())
    graph_builder.add_node("answer_generator", AnswerGenerator())

    graph_builder.set_entry_point("semantic_searcher")
    graph_builder.add_edge("semantic_searcher", "answer_generator")
    graph_builder.add_edge("answer_generator", END)

    return graph_builder.compile()


def run_rag_agent(user_input: str) -> Dict[str, str]:
    """Run the simple RAG graph and return a JSON-compatible output."""
    graph = build_rag_graph()
    result_state: RagState = graph.invoke({"user_input": user_input})
    return {"output": result_state["output"]}
