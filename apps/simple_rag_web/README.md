# simple_rag_web

Next.js client for `simple_rag_agent` HTTP API.

## Run

From the repository root:

```bash
cd apps/simple_rag_web
npm install
npm run dev
```

The app runs on `http://localhost:3000`.

## API Configuration

Set the backend base URL with:

```bash
NEXT_PUBLIC_RAG_API_URL=http://127.0.0.1:8000
```

If not set, the client defaults to `http://127.0.0.1:8000`.
