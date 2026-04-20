# Application Libraries

This document lists third-party Python libraries directly imported by the application code in this repository.

Scope:
- Includes `common/`, `simple_rag_agent/`, `custom_rag_agent/`, and `quickstart/`.
- Excludes `tests/`.
- Excludes Python standard library and local project modules.

## Core Runtime and API

- `fastapi`
  - Used in: `simple_rag_agent/api.py`, `custom_rag_agent/api.py`
- `pydantic`
  - Used in: `simple_rag_agent/api.py`, `custom_rag_agent/api.py`
- `python-dotenv` (`dotenv`)
  - Used in: `simple_rag_agent/api.py`, `custom_rag_agent/api.py`, `quickstart/test01.py`, `quickstart/embedding01.py`, `quickstart/agent01.py`, `quickstart/vector_store_search01.py`

## LangChain / LangGraph Stack

- `langchain-core` (`langchain_core`)
  - Used in: `simple_rag_agent/rag_agent.py`, `custom_rag_agent/rag_agent.py`, `simple_rag_agent/pdf_loader.py`, `custom_rag_agent/pdf_loader.py`, `simple_rag_agent/fake_knowledge_base.py`, `custom_rag_agent/fake_knowledge_base.py`, `quickstart/agent01.py`, `quickstart/vector_store_search01.py`
- `langgraph`
  - Used in: `simple_rag_agent/rag_agent.py`, `custom_rag_agent/rag_agent.py`, `quickstart/agent01.py`
- `langchain-text-splitters`
  - Used in: `simple_rag_agent/pdf_loader.py`, `custom_rag_agent/pdf_loader.py`
- `langchain-oci` (`langchain_oci`)
  - Used in: `common/oci_models.py`, `quickstart/test01.py`

## Document Processing and Utilities

- `pypdf`
  - Used in: `simple_rag_agent/pdf_loader.py`, `custom_rag_agent/pdf_loader.py`
- `tqdm`
  - Used in: `simple_rag_agent/pdf_loader.py`, `custom_rag_agent/pdf_loader.py`

## OpenAI-Compatible OCI Access

- `openai`
  - Used in: `common/oci_openai_clients.py`
- `httpx`
  - Used in: `common/oci_openai_clients.py`
- `oci-genai-auth` (`oci_genai_auth`)
  - Used in: `common/oci_openai_clients.py`
