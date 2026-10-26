"""
P5.4 — compare.py: runs react vs graph on routing, SQL, and retrieval dense vs hybrid;
writes eval/report.md with a summary table + latency percentiles + LLM calls per query.

Usage:
    cd backend && python eval/compare.py
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(len(values) * p / 100)
    return values[min(idx, len(values) - 1)]


# ── Routing (graph) ───────────────────────────────────────────────────────────

def _routing_graph(cases, services, llm, settings) -> dict:
    from app.graph.nodes.understand import make_understand
    from app.graph.state import turn_input

    understand = make_understand(llm, services)
    hits = total = 0
    latencies: list[float] = []
    llm_calls_total = 0

    for case in cases:
        t0 = time.perf_counter()
        state = turn_input(case["query"])
        result = asyncio.run(understand(state))
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        plan = result.get("plan")
        got = {s.source for s in plan.steps} if plan else set()
        expected = set(case["expected_sources"])
        ok = bool(got & expected)
        hits += int(ok)
        total += 1
        llm_calls_total += 1

    return {
        "mode": "graph", "metric": "routing",
        "hits": hits, "total": total,
        "accuracy": hits / total if total else 0,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "llm_calls_per_query": llm_calls_total / total if total else 0,
    }


# ── Retrieval (dense vs hybrid) ───────────────────────────────────────────────

def _retrieval_eval(cases, mode: str, embedder) -> dict:
    from app.tools.emails import EmailIndex, EmailSearchInput
    import json as _json

    emails_path = ROOT / "data" / "emails.json"
    with open(emails_path) as f:
        emails = _json.load(f)

    index = EmailIndex(emails, embedder, retrieval_mode=mode)
    hits = total = 0
    latencies: list[float] = []

    for case in cases:
        inp = EmailSearchInput(
            content_query=case.get("query"),
            sender=case.get("sender"),
            top_k=10,
        )
        t0 = time.perf_counter()
        try:
            result = index.search(inp)
        except ValueError:
            result = {"emails": []}
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        returned = {e["id"] for e in result["emails"]}
        expected = set(case.get("expected_ids", []))
        ok = len(returned & expected) >= case.get("expected_min", 1)
        hits += int(ok)
        total += 1

    return {
        "mode": mode, "metric": "retrieval",
        "hits": hits, "total": total,
        "accuracy": hits / total if total else 0,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "llm_calls_per_query": 0,
    }


# ── SQL ───────────────────────────────────────────────────────────────────────

def _sql_eval(cases, db_path: Path, llm, settings) -> dict:
    from app.tools.database import Database
    from app.graph.sql.subgraph import build_sql_subgraph

    db = Database(str(db_path))
    graph = build_sql_subgraph(db, llm, settings)
    hits = first_ok = total = 0
    latencies: list[float] = []
    llm_calls_total = 0

    for case in cases[:10]:  # run first 10 for speed
        t0 = time.perf_counter()
        result = asyncio.run(graph.ainvoke({
            "question": case["question"],
            "step_id": "s1",
            "context": "",
            "workspace_id": "",
            "attempts": 0,
            "signatures": [],
            "excluded_tables": [],
        }))
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        ok = result.get("status") in ("ok", "empty")
        hits += int(ok)
        first_ok += int(ok and result.get("first_try_ok", False))
        llm_calls_total += result.get("attempts", 1)
        total += 1

    return {
        "mode": "graph", "metric": "sql",
        "hits": hits, "total": total,
        "accuracy": hits / total if total else 0,
        "first_try_accuracy": first_ok / total if total else 0,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "llm_calls_per_query": llm_calls_total / total if total else 0,
    }


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(results: list[dict]) -> None:
    lines = [
        "# Eval Report\n",
        "Generated by `eval/compare.py`.\n",
        "| Metric | Mode | Accuracy | First-try | p50 ms | p95 ms | LLM calls/query |",
        "|--------|------|----------|-----------|--------|--------|-----------------|",
    ]
    for r in results:
        first_try = f"{r.get('first_try_accuracy', r['accuracy']):.0%}"
        lines.append(
            f"| {r['metric']} | {r['mode']} | {r['accuracy']:.0%} | {first_try} "
            f"| {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['llm_calls_per_query']:.2f} |"
        )
    out = ROOT / "eval" / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport written → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        print("SKIP: compare.py requires GOOGLE_API_KEY")
        return

    from app.config import settings
    from app.llm import make_llm
    from app.tools import Services
    from app.tools.emails import EmailIndex
    from app.tools.database import Database
    from app.embeddings import default_embedder
    import json as _json

    llm = make_llm(settings)

    emails_path = ROOT / "data" / "emails.json"
    with open(emails_path) as f:
        emails = _json.load(f)
    email_index = EmailIndex(emails, default_embedder)
    db_path = ROOT / "data" / "demo.db"
    database = Database(str(db_path)) if db_path.exists() else None
    services = Services(emails=email_index, database=database)

    results: list[dict] = []

    # Routing
    routing_cases = _load_jsonl(ROOT / "eval" / "routing_cases.jsonl")
    results.append(_routing_graph(routing_cases, services, llm, settings))
    print(f"Routing: {results[-1]['accuracy']:.0%}")

    # Retrieval: dense vs hybrid
    retrieval_cases = _load_jsonl(ROOT / "eval" / "email_retrieval_cases.jsonl")
    for mode in ("dense", "hybrid"):
        r = _retrieval_eval(retrieval_cases, mode, default_embedder)
        results.append(r)
        print(f"Retrieval ({mode}): {r['accuracy']:.0%}")

    # SQL
    if db_path.exists():
        sql_cases = _load_jsonl(ROOT / "eval" / "sql_cases.jsonl")
        r = _sql_eval(sql_cases, db_path, llm, settings)
        results.append(r)
        print(f"SQL: {r['accuracy']:.0%} ({r['first_try_accuracy']:.0%} first-try)")

    _write_report(results)


if __name__ == "__main__":
    main()
