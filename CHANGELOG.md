# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and uses semantic sections.

## [2026-04-19]

### Added
- Next.js web client example in `apps/simple_rag_web` for calling the Simple RAG HTTP API.
- Added `custom_rag_agent/` as a full baseline clone of `simple_rag_agent/`, including copied local `.env` and mirrored `tests/test_custom_*.py` for safe incremental evolution.
- Added streaming endpoint `POST /invoke/stream` in `custom_rag_agent/api.py` using Server-Sent Events (SSE) for step-by-step agent progress.
- Added `custom_rag_agent/stream_client.py` as a minimal Python CLI to test streaming events and token output.
- Added streaming tests in `tests/test_custom_rag_stream_client.py` and new streaming coverage in `tests/test_custom_rag_agent.py` and `tests/test_custom_rag_api.py`.
- Added `apps/custom_rag_web` as a dedicated Next.js streaming UI for `custom_rag_agent`.

### Changed
- Added a configuration sidebar to `apps/simple_rag_web` with editable backend invoke URL.
- Improved web client answer rendering by adding Markdown support (`react-markdown` + `remark-gfm`).
- Updated web client layout to place output below input and added loading spinner feedback during backend calls.
- Added UI-side conversation history tracking, forwarding `history` to backend API, and a sidebar button to clear history.
- Upgraded `custom_rag_agent/rag_agent.py` with `stream_rag_agent_events` to stream LangGraph step updates, semantic retrieval results, and final LLM token chunks.
- Updated `apps/custom_rag_web` to consume SSE events from `/invoke/stream`, show live step progress, render retrieved metadata in sidebar as soon as available, and stream final answer tokens in real time.
- Hardened query-rewrite behavior in both `simple_rag_agent` and `custom_rag_agent` so standalone search queries exclude explanations/comments and keep only retrievable query text.
- Simplified query-rewrite handling by removing post-processing sanitization and enforcing strict single-line standalone query output directly via prompt instructions.

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
