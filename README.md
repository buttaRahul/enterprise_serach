# Enterprise Search

An AI search solution for organisations — unified search across emails, structured databases, company websites, and the web. Built on **FastAPI + LangGraph** (backend) and **React + Vite + MUI** (frontend).

---

## Architecture

```
Browser (React + MUI)
    │  POST /api/chat/stream  (SSE)
    │  POST /api/chat          (batch)
    │  POST /api/feedback
    ▼
FastAPI  ──────────────────────────────────────────────────┐
    │                                                       │
    ▼                                                       │
LangGraph orchestration                                     │
  ┌──────────────┐                                          │
  │  preprocess  │  token estimate → needs_summary?         │
  └──────┬───────┘                                          │
         │                                                  │
  ┌──────▼───────┐  ┌──────────────┐                        │
  │  summarize   │  │  understand  │  Gemini → Plan/Intent  │
  └──────────────┘  └──────┬───────┘                        │
                           │                                │
              ┌────────────┼────────────┐                   │
              │            │            │                   │
         ┌────▼────┐  ┌────▼────┐  ┌───▼──────┐            │
         │ email   │  │   sql   │  │  site /  │            │
         │retrieve │  │subgraph │  │   web    │            │
         └────┬────┘  └────┬────┘  └───┬──────┘            │
              └────────────┴────────────┘                   │
                           │                                │
                    ┌──────▼───────┐                        │
                    │    reflect   │  retry / wave          │
                    └──────┬───────┘                        │
                           │                                │
                    ┌──────▼───────┐                        │
                    │  synthesize  │  Gemini → answer       │
                    └──────────────┘                        │
                                                            │
  Stores: SQLite checkpoints · SQLite feedback              │
  Retrieval: TF-IDF dense · BM25+dense hybrid (RRF)         │
  SQL guard: AST allowlist + forbidden columns + LIMIT      │
  SQL repair: classify → critic → regenerate (≤3 attempts)  │
└───────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph for orchestration | Explicit state machine with typed edges; easy to test nodes in isolation |
| Route on `Plan.intent` / `ToolResult.status`, never on LLM text | Prevents prompt-injection routing; all routing signals come from typed code |
| SQL subgraph with repair loop | Isolates SQL generation; classify → critic → regenerate keeps the main graph clean |
| Hybrid retrieval (BM25 + dense RRF) | Keyword recall for exact names/ids; dense recall for semantic queries |
| `make_safe` AST guard | Blocks writes, multi-statements, forbidden columns, unlisted tables before execution |
| Feedback store → few-shot SQL injection | Workspace-scoped positive examples and corrections improve SQL quality over time |
| TF-IDF fallback embedder | Avoids hard dependency on `sentence_transformers`/`torch` in CI |

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env
# edit .env — add GOOGLE_API_KEY=...

# Seed the demo database
python scripts/seed_demo_db.py

# Run tests (no API key or network required)
pytest -q

# Start the API server
uvicorn app.main:app --reload
# → http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend/llmapp
npm install
npm run dev
# → http://localhost:5173
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Liveness check |
| `POST` | `/api/chat` | Batch chat (returns full answer) |
| `POST` | `/api/chat/stream` | SSE streaming chat |
| `POST` | `/api/feedback` | Thumbs-up / thumbs-down + SQL correction |
| `POST` | `/api/search` | v1 ReAct alias (eval baseline) |

### Chat request

```json
{ "query": "How many engineers are in the company?", "thread_id": "optional-uuid" }
```

### Chat response

```json
{
  "thread_id": "...",
  "status": "answered",
  "answer": "There are 6 engineers...",
  "citations": ["e01", "e07"],
  "evidence": [{ "source": "sql", "rows": [...] }],
  "trace": [{ "node": "understand", "ms": 320 }]
}
```

---

## Eval

```bash
cd backend

# Routing accuracy
python eval/run_eval.py routing

# Email retrieval (dense vs hybrid)
python eval/run_eval.py retrieval

# SQL accuracy + latency percentiles
python eval/run_eval.py sql

# Multi-turn conversation
python eval/run_eval.py multiturn

# Full comparison: react vs graph, dense vs hybrid
python eval/compare.py
# → writes eval/report.md
```

---

## Project structure

```
backend/
  app/
    graph/          LangGraph nodes, SQL subgraph, state
    tools/          email index, database (make_safe), web, company site
    memory/         feedback store (SQLite)
    retrieval/      hybrid.py (BM25 + dense RRF)
    embeddings.py   sentence-transformers with TF-IDF fallback
    llm.py          LLM protocol + Gemini adapter
    main.py         FastAPI app (chat, feedback, SSE stream)
  tests/            72 offline tests (pytest -q)
  eval/             routing, retrieval, sql, multiturn eval harnesses
  data/             emails.json, demo.db, emails_large.json
  scripts/          seed_demo_db.py, make_synthetic_mailbox.py
frontend/
  llmapp/src/
    components/ChatBot.jsx   SSE streaming, citations, thumbs feedback
```
