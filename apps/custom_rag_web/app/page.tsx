"use client";

import { FormEvent, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type RetrievedDoc = {
  source?: string;
  title?: string;
  page?: number | string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type StepName = "query_rewriter" | "semantic_searcher" | "answer_generator";
type StepStatus = "pending" | "running" | "completed";

type StreamEvent = {
  event?: string;
  step?: StepName;
  message?: string;
  search_query?: string;
  token?: string;
  output?: string;
  retrieved_docs?: RetrievedDoc[];
};

type StepState = {
  name: StepName;
  label: string;
  status: StepStatus;
  info: string;
};

const FALLBACK_STREAM_URL = "http://127.0.0.1:8000/invoke/stream";

const INITIAL_STEPS: StepState[] = [
  {
    name: "query_rewriter",
    label: "Query Rewriter",
    status: "pending",
    info: "Waiting"
  },
  {
    name: "semantic_searcher",
    label: "Semantic Searcher",
    status: "pending",
    info: "Waiting"
  },
  {
    name: "answer_generator",
    label: "Answer Generator",
    status: "pending",
    info: "Waiting"
  }
];

function parseSseDataLine(line: string): StreamEvent | null {
  if (!line.startsWith("data:")) {
    return null;
  }

  const payloadRaw = line.slice(5).trim();
  if (!payloadRaw) {
    return null;
  }

  try {
    return JSON.parse(payloadRaw) as StreamEvent;
  } catch {
    return null;
  }
}

export default function HomePage() {
  const [question, setQuestion] = useState(
    "What are scaling laws?"
  );

  const defaultStreamUrl = useMemo(() => {
    const explicitStreamUrl = process.env.NEXT_PUBLIC_RAG_STREAM_URL?.trim();
    if (explicitStreamUrl) {
      return explicitStreamUrl;
    }

    const baseUrl = process.env.NEXT_PUBLIC_RAG_API_URL?.trim();
    if (baseUrl) {
      return `${baseUrl.replace(/\/$/, "")}/invoke/stream`;
    }

    return FALLBACK_STREAM_URL;
  }, []);

  const [streamUrl, setStreamUrl] = useState(defaultStreamUrl);
  const [topK, setTopK] = useState("4");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [answer, setAnswer] = useState("");
  const [docs, setDocs] = useState<RetrievedDoc[]>([]);
  const [steps, setSteps] = useState<StepState[]>(INITIAL_STEPS);
  const [searchQuery, setSearchQuery] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  function resetRuntimePanels() {
    setAnswer("");
    setDocs([]);
    setSteps(INITIAL_STEPS);
    setSearchQuery("");
  }

  function updateStep(stepName: StepName, status: StepStatus, info: string) {
    setSteps((previous) =>
      previous.map((step) => {
        if (step.name !== stepName) {
          return step;
        }
        return {
          ...step,
          status,
          info
        };
      })
    );
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsLoading(true);

    const trimmedQuestion = question.trim();
    const parsedTopK = Number.parseInt(topK, 10);
    const historySnapshot = [...history];
    resetRuntimePanels();

    try {
      if (!Number.isInteger(parsedTopK) || parsedTopK < 1) {
        throw new Error("top_k must be a positive integer.");
      }

      const response = await fetch(streamUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream"
        },
        body: JSON.stringify({
          request: trimmedQuestion,
          history: historySnapshot,
          top_k: parsedTopK
        })
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status} ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error("Streaming response body is not available.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let finalOutput = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";

        for (const line of lines) {
          const payload = parseSseDataLine(line.trim());
          if (!payload || !payload.event) {
            continue;
          }

          if (payload.event === "step_started" && payload.step) {
            updateStep(payload.step, "running", payload.message || "Running");
            continue;
          }

          if (payload.event === "step_completed" && payload.step) {
            let info = "Completed";
            if (payload.step === "query_rewriter" && payload.search_query) {
              info = `Search query: ${payload.search_query}`;
              setSearchQuery(payload.search_query);
            }
            if (payload.step === "semantic_searcher") {
              const count = payload.retrieved_docs?.length || 0;
              info = `Retrieved ${count} documents`;
            }
            updateStep(payload.step, "completed", info);
            continue;
          }

          if (payload.event === "retrieval_results") {
            setDocs(payload.retrieved_docs || []);
            continue;
          }

          if (payload.event === "final_answer_token") {
            const token = payload.token || "";
            if (token) {
              finalOutput += token;
              setAnswer((previous) => previous + token);
            }
            continue;
          }

          if (payload.event === "completed") {
            const completedOutput = payload.output || finalOutput;
            setAnswer(completedOutput);
            setDocs(payload.retrieved_docs || []);
            setHistory([
              ...historySnapshot,
              { role: "user", content: trimmedQuestion },
              { role: "assistant", content: completedOutput }
            ]);
          }
        }
      }
    } catch (submitError) {
      resetRuntimePanels();
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Unexpected error while calling the streaming API."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card reveal-up">
        <p className="eyebrow">OCI Enterprise AI</p>
        <h1>Custom RAG Streaming Web Client</h1>
        <p className="hero-description">
          Stream live agent progress, semantic retrieval metadata, and token-by-token
          answer generation from the custom RAG backend.
        </p>
      </section>

      <section className="workspace-grid">
        <aside className="panel settings-panel reveal-up delay-1">
          <h2>Configuration</h2>
          <label htmlFor="stream-url">Backend Stream URL</label>
          <input
            id="stream-url"
            type="url"
            value={streamUrl}
            onChange={(event) => setStreamUrl(event.target.value)}
            placeholder="http://127.0.0.1:8000/invoke/stream"
            required
          />
          <label htmlFor="top-k">top_k</label>
          <input
            id="top-k"
            type="number"
            min={1}
            step={1}
            value={topK}
            onChange={(event) => setTopK(event.target.value)}
            required
          />
          <p className="history-indicator">History messages: {history.length}</p>
          <div className="actions settings-actions">
            <button
              type="button"
              onClick={() => setStreamUrl(defaultStreamUrl)}
              disabled={streamUrl === defaultStreamUrl}
            >
              Reset URL
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={() => setHistory([])}
              disabled={history.length === 0}
            >
              Clear History
            </button>
          </div>

          <h2>Execution Progress</h2>
          <ul className="step-list">
            {steps.map((step) => (
              <li key={step.name}>
                <span className={`step-status ${step.status}`}>{step.status}</span>
                <div>
                  <p className="step-label">{step.label}</p>
                  <p className="step-info">{step.info}</p>
                </div>
              </li>
            ))}
          </ul>

          <h2>Retrieved Documents</h2>
          {docs.length === 0 ? (
            <p className="empty-state">No documents retrieved yet.</p>
          ) : (
            <ul className="doc-list sidebar-doc-list">
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
                disabled={isLoading || !question.trim() || !streamUrl.trim()}
              >
                {isLoading ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    Streaming...
                  </>
                ) : (
                  "Ask Streaming API"
                )}
              </button>
            </div>
            {error ? <p className="error-box">{error}</p> : null}
          </form>

          <section className="panel reveal-up delay-2">
            <div className="answer-header">
              <h2>Answer</h2>
              {isLoading ? (
                <span className="loading-chip">
                  <span className="spinner" aria-hidden="true" />
                  Receiving tokens...
                </span>
              ) : null}
            </div>
            {searchQuery ? (
              <p className="search-query-box">
                <strong>Search Query:</strong> {searchQuery}
              </p>
            ) : null}
            <div className="answer-box">
              {answer ? (
                <div className="markdown-output">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                </div>
              ) : (
                "No answer yet. Submit a question to start streaming."
              )}
            </div>
          </section>
        </section>
      </section>
    </main>
  );
}
