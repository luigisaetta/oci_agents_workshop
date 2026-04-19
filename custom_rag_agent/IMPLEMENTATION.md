# Custom RAG Agent: Detailed Implementation Guide

This document explains the full implementation of `custom_rag_agent`, including
module responsibilities, runtime flow, and the streaming protocol used by the UI.

## Scope and Goal

`custom_rag_agent` is a practical RAG implementation that provides:

- retrieval over a vector store (in-memory by default);
- optional PDF ingestion and indexing at API startup;
- synchronous invocation (`/invoke`);
- streaming invocation (`/invoke/stream`) with incremental progress and token events.

The implementation is intentionally modular so each concern is isolated in a
small file.

## Package Structure

- `custom_rag_agent/__init__.py`
- `custom_rag_agent/rag_agent.py`
- `custom_rag_agent/api.py`
- `custom_rag_agent/pdf_loader.py`
- `custom_rag_agent/prompts.py`
- `custom_rag_agent/fake_knowledge_base.py`
- `custom_rag_agent/client.py`
- `custom_rag_agent/stream_client.py`

The package also relies on shared helpers in:

- `common/utils.py`
- `common/oci_models.py`

## Core Orchestration (`rag_agent.py`)

File: `custom_rag_agent/rag_agent.py`

This is the central runtime module. It contains:

- state definitions (`RagState`, `RagStreamEvent`);
- runtime configuration assembly (`_collect_rag_runtime_config`);
- graph nodes (`QueryRewriter`, `SemanticSearcher`, `AnswerGenerator`);
- graph builders (`build_retrieval_graph`, `build_rag_graph`);
- execution entrypoints (`run_rag_agent`, `stream_rag_agent_events`).

### Shared State

`RagState` carries data across graph nodes, including:

- `user_input` and optional `history`;
- `runtime_config`;
- rewritten `search_query`;
- retrieved `documents`;
- final `output` and `retrieved_docs` metadata.

### Node 1: QueryRewriter

Class: `QueryRewriter`

Responsibilities:

- if no history is provided, reuse `user_input` directly;
- if history exists, call the LLM with a rewrite prompt from `prompts.py`;
- output a single rewritten `search_query` to improve retrieval quality.

### Node 2: SemanticSearcher

Class: `SemanticSearcher`

Responsibilities:

- read `SIMPLE_RAG_TOP_K` from runtime configuration;
- run vector similarity search against the configured vector store;
- store top-k documents in `state["documents"]`.

### Node 3: AnswerGenerator

Class: `AnswerGenerator`

Responsibilities:

- build context from retrieved document content;
- call the LLM with answer prompt;
- return final text output and metadata-only document list.

### Sync Path

Function: `run_rag_agent`

Flow:

1. Build full graph (`query_rewriter -> semantic_searcher -> answer_generator`).
2. Invoke graph once with full state.
3. Return JSON-compatible payload: `output`, `retrieved_docs`.

### Streaming Path

Function: `stream_rag_agent_events`

Flow:

1. Run retrieval-only graph in streaming updates mode.
2. Emit step progress events while rewrite/search complete.
3. Emit `retrieval_results` as soon as semantic search finishes.
4. Start answer generation by streaming LLM tokens.
5. Emit final `completed` event with consolidated output + metadata.

This design allows clients to show progress and retrieved metadata before the
final answer is completed.

## API Layer (`api.py`)

File: `custom_rag_agent/api.py`

Responsibilities:

- create and configure FastAPI app;
- initialize vector store once at startup (`lifespan`);
- expose synchronous and streaming routes.

### Startup Vector Store Strategy

At startup:

- if PDFs are present in `input_pdf/`, call `build_pdf_vector_store`;
- otherwise, call `build_initialized_vector_store` (fake KB fallback).

The resulting store is cached in `app.state.vector_store` and reused for all
requests.

### Endpoints

- `POST /invoke`
  - calls `run_rag_agent`;
  - returns classic JSON response.

- `POST /invoke/stream`
  - calls `stream_rag_agent_events`;
  - wraps events as SSE (`text/event-stream`) with `data: ...` lines.

### Streaming Event Types

Current event payloads include:

- `step_started`
- `step_completed`
- `retrieval_results`
- `final_answer_token`
- `completed`

## PDF Ingestion and Indexing (`pdf_loader.py`)

File: `custom_rag_agent/pdf_loader.py`

Responsibilities:

- detect PDF files in `input_pdf/`;
- extract text page by page with metadata (`source`, `title`, `page`);
- split pages into chunks (default `chunk_size=800`, `chunk_overlap=200`);
- embed chunks and load them into `InMemoryVectorStore` in batches.

This module is used at API startup when local PDFs are available.

## Prompt Templates (`prompts.py`)

File: `custom_rag_agent/prompts.py`

Provides two template builders:

- `build_query_rewrite_prompt`
- `build_answer_prompt`

The rewrite prompt is strict and requests exactly one standalone query line,
without additional comments, labels, or tags.

## Fake Knowledge Base (`fake_knowledge_base.py`)

File: `custom_rag_agent/fake_knowledge_base.py`

Provides a stable fallback corpus when no local PDFs are indexed.
Each item is converted into a `Document` with source metadata.

## CLI Clients

### Synchronous Client (`client.py`)

File: `custom_rag_agent/client.py`

- sends `POST /invoke`;
- prints final answer + retrieved document metadata.

### Streaming Client (`stream_client.py`)

File: `custom_rag_agent/stream_client.py`

- sends `POST /invoke/stream` with `Accept: text/event-stream`;
- parses SSE `data:` frames;
- prints step progress and token-level output in real time.

## Shared Dependencies (`common/*`)

### OCI Runtime and Utilities

File: `common/utils.py`

Key responsibilities:

- collect OCI runtime config from environment variables;
- extract plain text from heterogeneous model response formats.

### OCI Model Builders

File: `common/oci_models.py`

Key responsibilities:

- build chat LLM client (`build_llm`);
- build embedding client (`build_embedding_client`).

## Test Coverage

Relevant tests for this package are mainly in:

- `tests/test_custom_rag_agent.py`
- `tests/test_custom_rag_api.py`
- `tests/test_custom_pdf_loader.py`
- `tests/test_custom_fake_knowledge_base.py`
- `tests/test_custom_rag_client.py`
- `tests/test_custom_rag_stream_client.py`

They validate node behavior, graph output, API routes, startup branching,
streaming event protocol, and client-side stream parsing.

## End-to-End Runtime Sequence

Typical streaming request sequence:

1. UI/client sends `POST /invoke/stream` with `request` + optional `history`.
2. API routes request to `stream_rag_agent_events`.
3. Query rewrite step starts/completes.
4. Semantic retrieval step starts/completes.
5. Retrieval metadata is emitted (`retrieval_results`).
6. Answer generation starts.
7. Tokens stream progressively (`final_answer_token`).
8. Final payload closes the sequence (`completed`).

This sequence is what enables real-time progress indicators and early retrieval
visibility in UI applications.
