# Quickstart Programs

This folder contains runnable examples to validate OCI access and
understand the workshop building blocks.

## Programs

- `quickstart/test01.py`: minimal streaming chat call to OCI Generative AI.
- `quickstart/agent01.py`: simple 3-step LangGraph agent that returns JSON output.
- `quickstart/embedding01.py`: embedding generation example with compact vector summary.
- `quickstart/vector_store_search01.py`: semantic vector store search using OpenAI SDK, with output compatible with `custom_rag_agent` retrieved docs metadata.

## Run From Project Root

Run these commands from the repository root:

```bash
python -m quickstart.test01
python -m quickstart.agent01 "Explain what a Dedicated AI Cluster is in OCI."
python -m quickstart.embedding01
python -m quickstart.vector_store_search01 "What are scaling laws for LLMs?"
```

Equivalent path-based commands also work:

```bash
python quickstart/test01.py
python quickstart/agent01.py "Explain what a Dedicated AI Cluster is in OCI."
python quickstart/embedding01.py
python quickstart/vector_store_search01.py "What are scaling laws for LLMs?"
```

For `vector_store_search01.py`, set these env vars in `.env`:
- `OCI_OPENAI_BASE_URL`
- `OCI_OPENAI_PROJECT_ID`
- `OCI_VECTOR_STORE_ID`
