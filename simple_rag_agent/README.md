# Simple RAG Agent

This folder contains a minimal LangGraph RAG example with two Runnable steps:

- `SemanticSearcher`
- `AnswerGenerator`

The API returns a JSON payload with two fields:

- `output`
- `retrieved_docs` (metadata only, including `source`, `title`, and `page` when available)

When the API starts, it loads and indexes PDF chunks from `input_pdf/` when
available. If no PDF files are present, it falls back to the fake knowledge
base. Query requests reuse this pre-indexed store.

## Requirements

- OCI configuration in `$HOME/.oci/config`
- `.env` file in project root
- `OCI_EMBED_MODEL_ID` set in `.env`
- `SIMPLE_RAG_TOP_K` set in `.env` (optional, default is `4`)
- `pypdf` installed in your Python environment
- `langchain-text-splitters` installed for chunking
- `tqdm` installed for embedding progress bar
- `input_pdf/` folder under project root for source PDF files

The loader script uses:

- `chunk_size=800`
- `chunk_overlap=200`

## Run the API server

From project root:

```bash
uvicorn simple_rag_agent.api:app --host 127.0.0.1 --port 8000 --reload
```

Before starting the API, put your PDF files into `input_pdf/`.
If no PDF is found, the API falls back to the fake knowledge base.

## Run the client

From project root:

```bash
python simple_rag_agent/client.py "What are scaling laws for LLM?"
```

Optional custom URL:

```bash
python simple_rag_agent/client.py "Your question" --url http://127.0.0.1:8000/invoke
```
