"use client";

import { FormEvent, useMemo, useState } from "react";

type RetrievedDoc = {
  source?: string;
  title?: string;
  page?: number | string;
};

type InvokeResponse = {
  output: string;
  retrieved_docs: RetrievedDoc[];
};

const FALLBACK_INVOKE_URL = "http://127.0.0.1:8000/invoke";

export default function HomePage() {
  const [question, setQuestion] = useState(
    "What is a Dedicated AI Cluster in OCI?"
  );
  const defaultInvokeUrl = useMemo(() => {
    const explicitInvokeUrl = process.env.NEXT_PUBLIC_RAG_INVOKE_URL?.trim();
    if (explicitInvokeUrl) {
      return explicitInvokeUrl;
    }

    const baseUrl = process.env.NEXT_PUBLIC_RAG_API_URL?.trim();
    if (baseUrl) {
      return `${baseUrl.replace(/\/$/, "")}/invoke`;
    }

    return FALLBACK_INVOKE_URL;
  }, []);

  const [invokeUrl, setInvokeUrl] = useState(defaultInvokeUrl);
  const [answer, setAnswer] = useState("");
  const [docs, setDocs] = useState<RetrievedDoc[]>([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const response = await fetch(invokeUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ request: question })
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status} ${response.statusText}`);
      }

      const payload = (await response.json()) as InvokeResponse;
      setAnswer(payload.output || "");
      setDocs(payload.retrieved_docs || []);
    } catch (submitError) {
      setAnswer("");
      setDocs([]);
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Unexpected error while calling the API."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card reveal-up">
        <p className="eyebrow">OCI Enterprise AI</p>
        <h1>Simple RAG Web Client</h1>
        <p className="hero-description">
          Ask a question, call the FastAPI endpoint, and inspect both the final
          answer and retrieved document metadata.
        </p>
        <p className="endpoint-pill">
          Endpoint: <code>{invokeUrl}</code>
        </p>
      </section>

      <section className="workspace-grid">
        <aside className="panel settings-panel reveal-up delay-1">
          <h2>Configuration</h2>
          <label htmlFor="invoke-url">Backend Invoke URL</label>
          <input
            id="invoke-url"
            type="url"
            value={invokeUrl}
            onChange={(event) => setInvokeUrl(event.target.value)}
            placeholder="http://127.0.0.1:8000/invoke"
            required
          />
          <p className="settings-help">
            Default: <code>{FALLBACK_INVOKE_URL}</code>
          </p>
          <div className="actions">
            <button
              type="button"
              onClick={() => setInvokeUrl(defaultInvokeUrl)}
              disabled={invokeUrl === defaultInvokeUrl}
            >
              Reset URL
            </button>
          </div>
        </aside>

        <section className="content-grid">
          <form className="panel reveal-up delay-2" onSubmit={onSubmit}>
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask something about OCI, DAC, or your indexed docs..."
              rows={6}
              required
            />
            <div className="actions">
              <button
                type="submit"
                disabled={isLoading || !question.trim() || !invokeUrl.trim()}
              >
                {isLoading ? "Sending..." : "Ask RAG API"}
              </button>
            </div>
            {error ? <p className="error-box">{error}</p> : null}
          </form>

          <section className="panel reveal-up delay-2">
            <h2>Answer</h2>
            <div className="answer-box">
              {answer || "No answer yet. Submit a question to start."}
            </div>

            <h2>Retrieved Documents</h2>
            {docs.length === 0 ? (
              <p className="empty-state">No documents returned yet.</p>
            ) : (
              <ul className="doc-list">
                {docs.map((doc, index) => (
                  <li key={`${doc.source || "doc"}-${index}`}>
                    <span className="doc-index">{index + 1}</span>
                    <div>
                      <p>
                        <strong>Source:</strong> {doc.source || "unknown"}
                      </p>
                      <p>
                        <strong>Title:</strong> {doc.title || "n/a"}
                      </p>
                      <p>
                        <strong>Page:</strong> {doc.page ?? "n/a"}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </section>
      </section>
    </main>
  );
}
