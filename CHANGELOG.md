# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and uses semantic sections.

## [Unreleased]

### Added
- Next.js web client example in `apps/simple_rag_web` for calling the Simple RAG HTTP API.

### Changed
- Added a configuration sidebar to `apps/simple_rag_web` with editable backend invoke URL.
- Improved web client answer rendering by adding Markdown support (`react-markdown` + `remark-gfm`).
- Updated web client layout to place output below input and added loading spinner feedback during backend calls.
- Added UI-side conversation history tracking, forwarding `history` to backend API, and a sidebar button to clear history.

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
