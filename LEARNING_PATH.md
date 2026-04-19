# Learning Path: From First OCI Call to Streaming RAG

This guide proposes a practical, progressive study path for this repository.
It is designed for learners who want to move from foundational OCI usage to a
streaming RAG application with step-by-step observability.

## How to Use This Guide

- Follow the stages in order.
- Run each example before moving to the next one.
- Inspect the linked files to understand how each capability is implemented.

## Recommended Study Sequence

### Stage 1: Environment and OCI Runtime Basics

1. Read [quickstart/README.md](./quickstart/README.md).
2. Run [quickstart/test01.py](./quickstart/test01.py) to validate OCI config.

What you learn:

- required environment variables;
- OCI runtime configuration and basic model connectivity;
- first streamed model output.

### Stage 2: Basic Agent Invocation

1. Run [quickstart/agent01.py](./quickstart/agent01.py).

What you learn:

- minimal prompt-to-response flow;
- simple agent-style orchestration from CLI.

### Stage 3: Embeddings Fundamentals

1. Run [quickstart/embedding01.py](./quickstart/embedding01.py).

What you learn:

- embedding generation;
- why embeddings are the basis for vector retrieval in RAG.

### Stage 4: First RAG Pipeline (Synchronous)

1. Read [simple_rag_agent/README.md](./simple_rag_agent/README.md).
2. Inspect [simple_rag_agent/rag_agent.py](./simple_rag_agent/rag_agent.py).
3. Start [simple_rag_agent/api.py](./simple_rag_agent/api.py) and call it with
   [simple_rag_agent/client.py](./simple_rag_agent/client.py).

What you learn:

- query rewrite, semantic search, and answer generation as separate steps;
- API-first packaging of a RAG flow;
- metadata returned with retrieved documents.

### Stage 5: PDF-Based Indexing

1. Inspect [simple_rag_agent/pdf_loader.py](./simple_rag_agent/pdf_loader.py).
2. Put PDFs in `input_pdf/` and restart the API.

What you learn:

- document loading and page metadata extraction;
- chunking strategy and embedding/indexing pipeline;
- switching from fake KB to local document retrieval.

### Stage 6: Web UI for Synchronous RAG

1. Read [apps/simple_rag_web/README.md](./apps/simple_rag_web/README.md).
2. Run the UI and connect it to `simple_rag_agent`.

What you learn:

- integrating a Next.js client with a FastAPI RAG backend;
- request/response UX with chat history forwarding.

### Stage 7: Advanced RAG with Streaming

1. Read [custom_rag_agent/README.md](./custom_rag_agent/README.md).
2. Study [custom_rag_agent/rag_agent.py](./custom_rag_agent/rag_agent.py) and
   [custom_rag_agent/api.py](./custom_rag_agent/api.py).
3. Run streaming CLI: [custom_rag_agent/stream_client.py](./custom_rag_agent/stream_client.py).
4. Read the technical deep dive:
   [custom_rag_agent/IMPLEMENTATION.md](./custom_rag_agent/IMPLEMENTATION.md).

What you learn:

- LangGraph update streaming for intermediate step progress;
- SSE protocol design for UI streaming;
- token-level final answer streaming.

### Stage 8: Streaming Web UI and Real-Time Observability

1. Read [apps/custom_rag_web/README.md](./apps/custom_rag_web/README.md).
2. Run the UI against `POST /invoke/stream`.

What you learn:

- live progress visualization by agent step;
- early display of retrieval metadata;
- real-time token streaming in the final answer panel.

## Example Index (Simple to Complex)

| Level | Example | What It Shows |
|---|---|---|
| 1 | [quickstart/test01.py](./quickstart/test01.py) | OCI runtime config check and first streamed model output. |
| 2 | [quickstart/agent01.py](./quickstart/agent01.py) | Minimal single-agent invocation flow from CLI. |
| 3 | [quickstart/embedding01.py](./quickstart/embedding01.py) | Embedding generation basics for retrieval use cases. |
| 4 | [simple_rag_agent/fake_knowledge_base.py](./simple_rag_agent/fake_knowledge_base.py) | Simple fallback corpus used when no PDFs are indexed. |
| 5 | [simple_rag_agent/rag_agent.py](./simple_rag_agent/rag_agent.py) | Three-step RAG graph: rewrite, retrieve, answer. |
| 6 | [simple_rag_agent/api.py](./simple_rag_agent/api.py) | FastAPI wrapper around synchronous RAG invocation. |
| 7 | [simple_rag_agent/client.py](./simple_rag_agent/client.py) | CLI caller for `/invoke` and formatted output rendering. |
| 8 | [simple_rag_agent/pdf_loader.py](./simple_rag_agent/pdf_loader.py) | PDF parsing, chunking, embedding, and vector indexing. |
| 9 | [apps/simple_rag_web](./apps/simple_rag_web) | Next.js UI for synchronous RAG interaction. |
| 10 | [custom_rag_agent/rag_agent.py](./custom_rag_agent/rag_agent.py) | Advanced RAG with event streaming and token streaming orchestration. |
| 11 | [custom_rag_agent/api.py](./custom_rag_agent/api.py) | SSE endpoint (`/invoke/stream`) for real-time client updates. |
| 12 | [custom_rag_agent/stream_client.py](./custom_rag_agent/stream_client.py) | Terminal client consuming SSE events and token output. |
| 13 | [apps/custom_rag_web](./apps/custom_rag_web) | Next.js streaming UI with live step progress and retrieval metadata. |
