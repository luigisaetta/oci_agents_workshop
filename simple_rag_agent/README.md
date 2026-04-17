# Simple RAG Agent

This folder contains a minimal LangGraph RAG example with two Runnable steps:

- `SemanticSearcher`
- `AnswerGenerator`

The API returns a JSON payload with two fields:

- `output`
- `retrieved_docs`

When the API starts, it loads the fake knowledge base and builds the in-memory
vector index once. Query requests reuse this pre-indexed store.

## Requirements

- OCI configuration in `$HOME/.oci/config`
- `.env` file in project root
- `OCI_EMBED_MODEL_ID` set in `.env`
- `SIMPLE_RAG_TOP_K` set in `.env` (optional, default is `4`)

## Run the API server

From project root:

```bash
uvicorn simple_rag_agent.api:app --host 127.0.0.1 --port 8000 --reload
```

## Run the client

From project root:

```bash
python simple_rag_agent/client.py "What is a Dedicated AI Cluster in OCI?"
```

Optional custom URL:

```bash
python simple_rag_agent/client.py "Your question" --url http://127.0.0.1:8000/invoke
```
