# CLAUDE.md

Enterprise Search: FastAPI + LangGraph backend (`backend/`), React + Vite + MUI frontend (`frontend/llmapp/`).

## Current work
Implementing v2 (the LangGraph orchestration layer) from `docs/v2/SPEC.md`. Work through its task list in order (P0 → P5). Read SPEC §0–§4 before any task.

## Rules
- **Commits:** imperative subject, one commit per spec task. **Never add `Co-Authored-By`, `Claude-Session` or any AI-attribution trailer.**
- **Clean-room:** don't use code, prompts, names or domain logic from any employer codebase. Nothing healthcare-, telephony- or customer-specific.
- **Tests offline:** `cd backend && pytest -q` must pass with no network or API keys. Inject LLMs and embedders; use `ScriptedLLM`, `ScriptedChatModel` and `hashing_embedder` from `tests/conftest.py`.
- **Keep v1:** `app/agent.py` (ReAct) is the eval baseline behind `AGENT_MODE=react`.
- **Typed boundaries:** route on `ToolResult.status` / `error_kind` / `Plan.intent`, never on text. Trusted values come from `config["configurable"]`, never from model output.
- **SQL:** every query goes through `make_safe` (AST guard + allowlist + LIMIT) on a read-only connection.

## Commands
```bash
cd backend
pip install -r requirements.txt
pytest -q
python scripts/seed_demo_db.py
uvicorn app.main:app --reload             # needs GOOGLE_API_KEY in backend/.env
python eval/run_eval.py retrieval|routing|sql|multiturn
cd ../frontend/llmapp && npm install && npm run dev && npx eslint src
```
