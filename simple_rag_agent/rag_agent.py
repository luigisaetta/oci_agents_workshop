"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Minimal two-step LangGraph RAG agent using OCI embeddings and LLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable
from langgraph.graph import END, StateGraph

from oci_models import build_embedding_client, build_llm
from simple_rag_agent.in_memory_vector_store import InMemoryVectorStore
from simple_rag_agent.prompts import build_answer_prompt
from utils import collect_oci_runtime_config, extract_text


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


# pylint: disable=too-few-public-methods
class SemanticSearcher(RunnableSerializable[RagState, RagState]):
    """Step 1: retrieve most relevant documents with semantic similarity."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        self._vector_store = vector_store or InMemoryVectorStore()

    def invoke(self, state: RagState, _config: Any = None, **_kwargs: Any) -> RagState:
        """Retrieve top relevant documents for the user request."""
        logging.info("START SemanticSearcher")

        runtime_config = _collect_rag_runtime_config()
        embedding_client = build_embedding_client(runtime_config)

        top_documents = self._vector_store.search(
            query=state["user_input"],
            embedding_client=embedding_client,
        )

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
