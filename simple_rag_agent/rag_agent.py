"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Minimal three-step LangGraph RAG agent with query rewrite support.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, StateGraph

from common.utils import collect_oci_runtime_config, extract_text
from common.oci_models import build_embedding_client, build_llm
from simple_rag_agent.fake_knowledge_base import build_fake_documents
from simple_rag_agent.prompts import build_answer_prompt, build_query_rewrite_prompt


class RagState(TypedDict, total=False):
    """State shared by all RAG graph steps."""

    user_input: str
    history: List[Dict[str, str]]
    runtime_config: Dict[str, str]
    search_query: str
    documents: List[Document]
    retrieved_docs: List[Dict[str, Any]]
    output: str


def _collect_rag_runtime_config() -> Dict[str, str]:
    """Collect runtime settings required by the simple RAG pipeline.

    Returns:
        Dict[str, str]: OCI runtime settings enriched with embedding model id
            and retrieval top-k value.

    Raises:
        ValueError: If required environment variables are missing or invalid.
    """
    runtime_config = collect_oci_runtime_config()
    embed_model_id = os.getenv("OCI_EMBED_MODEL_ID", "").strip()
    if not embed_model_id:
        raise ValueError("Set OCI_EMBED_MODEL_ID environment variable.")

    top_k_raw = os.getenv("SIMPLE_RAG_TOP_K", "4").strip()
    try:
        top_k = int(top_k_raw)
    except ValueError as error:
        raise ValueError("Set SIMPLE_RAG_TOP_K as a positive integer.") from error
    if top_k < 1:
        raise ValueError("Set SIMPLE_RAG_TOP_K as a positive integer.")

    runtime_config["OCI_EMBED_MODEL_ID"] = embed_model_id
    runtime_config["SIMPLE_RAG_TOP_K"] = str(top_k)
    return runtime_config


def build_initialized_vector_store(
    vector_store: InMemoryVectorStore | None = None,
) -> InMemoryVectorStore:
    """Build a vector store for retrieval when no external store is provided.

    Args:
        vector_store: Optional pre-built vector store to reuse as-is.

    Returns:
        InMemoryVectorStore: Retrieval-ready vector store.
            If ``vector_store`` is provided, it is returned unchanged.
            If not provided, a new in-memory store is created and seeded with
            fake knowledge-base documents.
    """
    logging.info("START VectorStoreLoadingAndIndexing")

    if vector_store is not None:
        logging.info("Using provided vector store without fake KB seeding.")
        logging.info("END VectorStoreLoadingAndIndexing")
        return vector_store

    runtime_config = _collect_rag_runtime_config()
    embedding_client = build_embedding_client(runtime_config)
    store = InMemoryVectorStore(embedding=embedding_client)

    base_documents = build_fake_documents()
    store.add_documents(base_documents)
    logging.info("Loaded documents (fake KB seed): %s", len(base_documents))
    logging.info("END VectorStoreLoadingAndIndexing")
    return store


# pylint: disable=too-few-public-methods
class QueryRewriter(RunnableSerializable[RagState, RagState]):
    """Step 1: rewrite retrieval query using conversation history."""

    def invoke(self, state: RagState, _config: Any = None, **_kwargs: Any) -> RagState:
        """Rewrite query if history exists, otherwise keep original user input.

        Args:
            state: Input graph state containing user input and optional history.
            _config: Optional LangGraph runtime config, unused.
            **_kwargs: Extra LangGraph invocation arguments, unused.

        Returns:
            RagState: Updated state including runtime config and search query.
        """
        logging.info("START QueryRewriter")
        runtime_config = _collect_rag_runtime_config()
        history = state.get("history", [])

        if not history:
            logging.info("No history provided; using raw user input for retrieval.")
            updated_state: RagState = dict(state)
            updated_state["runtime_config"] = runtime_config
            updated_state["search_query"] = state["user_input"]
            logging.info("END QueryRewriter")
            return updated_state

        llm = build_llm(runtime_config)
        rewrite_prompt = build_query_rewrite_prompt(
            user_input=state["user_input"],
            history=history,
        )
        rewritten_query = extract_text(llm.invoke(rewrite_prompt)).strip()

        updated_state = dict(state)
        updated_state["runtime_config"] = runtime_config
        updated_state["search_query"] = rewritten_query or state["user_input"]

        logging.info("END QueryRewriter")
        return updated_state


# pylint: disable=too-few-public-methods
class SemanticSearcher(RunnableSerializable[RagState, RagState]):
    """Step 2: retrieve most relevant documents with semantic similarity."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        self._vector_store = vector_store or build_initialized_vector_store()

    def invoke(self, state: RagState, _config: Any = None, **_kwargs: Any) -> RagState:
        """Retrieve top relevant documents for the user request.

        Args:
            state: Input graph state containing at least ``user_input``.
            _config: Optional LangGraph runtime config, unused.
            **_kwargs: Extra LangGraph invocation arguments, unused.

        Returns:
            RagState: Updated state including runtime config and retrieved docs.
        """
        logging.info("START SemanticSearcher")

        runtime_config = _collect_rag_runtime_config()
        top_k = int(runtime_config["SIMPLE_RAG_TOP_K"])
        search_query = state.get("search_query", state["user_input"])
        top_documents = self._vector_store.similarity_search(
            search_query,
            k=top_k,
        )

        updated_state: RagState = dict(state)
        updated_state["runtime_config"] = runtime_config
        updated_state["documents"] = top_documents

        logging.info("END SemanticSearcher")
        return updated_state


# pylint: disable=too-few-public-methods
class AnswerGenerator(RunnableSerializable[RagState, RagState]):
    """Step 3: generate final answer from retrieved documents."""

    def invoke(self, state: RagState, _config: Any = None, **_kwargs: Any) -> RagState:
        """Generate the final answer from retrieved document context.

        Args:
            state: Input graph state containing user input and retrieved docs.
            _config: Optional LangGraph runtime config, unused.
            **_kwargs: Extra LangGraph invocation arguments, unused.

        Returns:
            RagState: Updated state with final answer and document metadata.
        """
        logging.info("START AnswerGenerator")

        runtime_config = state.get("runtime_config", _collect_rag_runtime_config())
        llm = build_llm(runtime_config)

        # Flatten retrieved document content into a single context string.
        context = "\n\n".join(document.page_content for document in state["documents"])
        prompt = build_answer_prompt(user_input=state["user_input"], context=context)
        response = llm.invoke(prompt)
        retrieved_docs: List[Dict[str, Any]] = []
        for document in state["documents"]:
            retrieved_docs.append(dict(document.metadata))

        updated_state: RagState = dict(state)
        updated_state["output"] = extract_text(response)
        updated_state["retrieved_docs"] = retrieved_docs

        logging.info("END AnswerGenerator")
        return updated_state


def build_rag_graph(vector_store: InMemoryVectorStore | None = None):
    """Build and compile the simple three-step RAG graph.

    Args:
        vector_store: Optional pre-built vector store used by search step.

    Returns:
        Any: Compiled LangGraph runnable.
    """
    graph_builder = StateGraph(RagState)
    graph_builder.add_node("query_rewriter", QueryRewriter())
    graph_builder.add_node(
        "semantic_searcher",
        SemanticSearcher(vector_store=vector_store),
    )
    graph_builder.add_node("answer_generator", AnswerGenerator())

    graph_builder.set_entry_point("query_rewriter")
    graph_builder.add_edge("query_rewriter", "semantic_searcher")
    graph_builder.add_edge("semantic_searcher", "answer_generator")
    graph_builder.add_edge("answer_generator", END)

    return graph_builder.compile()


def run_rag_agent(
    user_input: str,
    history: List[Dict[str, str]] | None = None,
    vector_store: InMemoryVectorStore | None = None,
) -> Dict[str, Any]:
    """Run the RAG graph and return a JSON-compatible payload.

    Args:
        user_input: Natural language query from the caller.
        history: Optional list of prior conversation messages.
        vector_store: Optional pre-built vector store for retrieval.

    Returns:
        Dict[str, Any]: Dictionary with model output and retrieved docs metadata.
    """
    graph = build_rag_graph(vector_store=vector_store)

    # here we call the agent
    history_messages = history or []
    result_state: RagState = graph.invoke(
        {"user_input": user_input, "history": history_messages}
    )

    return {
        "output": result_state["output"],
        "retrieved_docs": result_state["retrieved_docs"],
    }
