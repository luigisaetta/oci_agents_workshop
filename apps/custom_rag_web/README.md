# custom_rag_web

Next.js streaming client for `custom_rag_agent` SSE API.

## Run

From the repository root:

```bash
cd apps/custom_rag_web
npm install
npm run dev
```

The app runs on `http://localhost:3000`.

## API Configuration

Set the streaming endpoint with:

```bash
NEXT_PUBLIC_RAG_STREAM_URL=http://127.0.0.1:8000/invoke/stream
```

If not set, the client defaults to `http://127.0.0.1:8000/invoke/stream`.

You can also edit the URL directly from the UI sidebar (`Configuration` panel).
