# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and uses semantic sections.

## [2026-04-20]

### Added
- Added `quickstart/responses01.py`, a minimal streaming Responses API example with CLI input and fixed model override to `openai.gpt-oss-120B` using the existing OCI OpenAI-compatible client.
- Added unit tests in `tests/test_responses01.py` for runtime config validation, streaming event handling, and CLI flow.

### Changed
- Added PDF filename logging at document-loading startup in both `simple_rag_agent/pdf_loader.py` and `custom_rag_agent/pdf_loader.py`.
- Simplified embedding-stage logging by removing per-batch filename logs while keeping progress bar behavior unchanged.
- Updated PDF loader tests in `tests/test_pdf_loader.py` and `tests/test_custom_pdf_loader.py` to reflect the simplified logging behavior.
- Updated `quickstart/README.md` with `responses01.py` usage and required environment variables.

## [2026-04-19]

### Added
- Next.js web client example in `apps/simple_rag_web` for calling the Simple RAG HTTP API.
- Added `custom_rag_agent/` as a full baseline clone of `simple_rag_agent/`, including copied local `.env` and mirrored `tests/test_custom_*.py` for safe incremental evolution.
- Added streaming endpoint `POST /invoke/stream` in `custom_rag_agent/api.py` using Server-Sent Events (SSE) for step-by-step agent progress.
- Added `custom_rag_agent/stream_client.py` as a minimal Python CLI to test streaming events and token output.
- Added streaming tests in `tests/test_custom_rag_stream_client.py` and new streaming coverage in `tests/test_custom_rag_agent.py` and `tests/test_custom_rag_api.py`.
- Added `apps/custom_rag_web` as a dedicated Next.js streaming UI for `custom_rag_agent`.
- Added `common/oci_openai_clients.py` with an OpenAI-compatible OCI client builder using `user_principal` authentication.
- Added `quickstart/vector_store_search01.py` to run semantic vector store search via OpenAI SDK and return metadata compatible with `custom_rag_agent`.
- Added unit tests `tests/test_openai_clients.py` and `tests/test_vector_store_search01.py`.

### Changed
- Added a configuration sidebar to `apps/simple_rag_web` with editable backend invoke URL.
- Improved web client answer rendering by adding Markdown support (`react-markdown` + `remark-gfm`).
- Updated web client layout to place output below input and added loading spinner feedback during backend calls.
- Added UI-side conversation history tracking, forwarding `history` to backend API, and a sidebar button to clear history.
- Upgraded `custom_rag_agent/rag_agent.py` with `stream_rag_agent_events` to stream LangGraph step updates, semantic retrieval results, and final LLM token chunks.
- Updated `apps/custom_rag_web` to consume SSE events from `/invoke/stream`, show live step progress, render retrieved metadata in sidebar as soon as available, and stream final answer tokens in real time.
- Hardened query-rewrite behavior in both `simple_rag_agent` and `custom_rag_agent` so standalone search queries exclude explanations/comments and keep only retrievable query text.
- Simplified query-rewrite handling by removing post-processing sanitization and enforcing strict single-line standalone query output directly via prompt instructions.
- Updated project dependencies to include `openai`, `httpx`, and `oci-genai-auth`.
- Simplified OpenAI-compatible client implementation by removing control-plane-specific client code and optional import fallbacks.

## [2026-04-18]

### Added
- `pyproject.toml` with editable installation support and `dev` extra dependencies.
- Explicit `Project Goal` section in `AGENTS.md`.
- `quickstart/README.md` for runnable quickstart examples.
- `quickstart/` and `common/` package layout to organize examples and reusable modules.

### Changed
- Repository refactoring: moved quickstart scripts into `quickstart/`.
- Repository refactoring: moved shared utilities into `common/`.
- README introduction expanded to describe end-to-end progression toward complex OCI agents.
- README setup updated with both `conda` and `venv` options.
- README Oracle section improved with additional reference to `oracle/langchain-oracle/samples`.
- Docstrings and inline comments improved across core modules for readability.
