"""
Offline-capable evaluation harness.

Usage:
    python eval/run_eval.py retrieval
    python eval/run_eval.py routing           # uses ScriptedLLM for offline, real LLM with GOOGLE_API_KEY
    AGENT_MODE=graph python eval/run_eval.py routing --mode graph
    python eval/run_eval.py sql
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure backend/ is on sys.path when run from anywhere
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Retrieval eval ────────────────────────────────────────────────────────────

def eval_retrieval(cases_path: str | None = None) -> None:
    from tests.conftest import hashing_embedder, SAMPLE_EMAILS
    from app.tools.emails import EmailIndex, EmailSearchInput

    path = cases_path or str(ROOT / "eval" / "email_retrieval_cases.jsonl")
    cases = [json.loads(l) for l in open(path) if l.strip()]

    index = EmailIndex(SAMPLE_EMAILS, hashing_embedder)
    hits = total = 0

    for case in cases:
        inp = EmailSearchInput(
            content_query=case.get("query"),
            sender=case.get("sender"),
            sent_before=case.get("sent_before"),
            sent_after=case.get("sent_after"),
            sent_on=case.get("sent_on"),
            top_k=10,
        )
        try:
            result = index.search(inp)
        except ValueError:
            result = {"emails": []}

        returned_ids = {e["id"] for e in result["emails"]}
        expected = set(case.get("expected_ids", []))
        min_hits = case.get("expected_min", 1)
        matched = len(returned_ids & expected)
        ok = matched >= min_hits
        hits += int(ok)
        total += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] {case['query']!r}: got {matched}/{len(expected)} expected")

    print(f"\nRetrieval: {hits}/{total} cases passed")


# ── Routing eval ──────────────────────────────────────────────────────────────

def eval_routing(mode: str = "react") -> None:
    from tests.conftest import hashing_embedder, SAMPLE_EMAILS
    from app.tools import Services
    from app.tools.emails import EmailIndex

    path = ROOT / "eval" / "routing_cases.jsonl"
    cases = [json.loads(l) for l in open(path) if l.strip()]

    email_index = EmailIndex(SAMPLE_EMAILS, hashing_embedder)
    services = Services(emails=email_index)

    if mode == "graph":
        _eval_routing_graph(cases, services)
    else:
        _eval_routing_react(cases, services)


def _eval_routing_graph(cases, services) -> None:
    """Score plan.steps[].source set vs expected (calls only understand, no execution)."""
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        print("SKIP: AGENT_MODE=graph routing eval requires GOOGLE_API_KEY")
        return

    from app.config import settings
    from app.llm import make_llm
    from app.graph.nodes.understand import make_understand
    from app.graph.state import turn_input

    llm = make_llm(settings)
    understand = make_understand(llm, services)

    async def _run(query: str) -> list[str]:
        state = turn_input(query)
        result = await understand(state)
        plan = result.get("plan")
        return [s.source for s in plan.steps] if plan else []

    hits = total = 0
    for case in cases:
        sources = asyncio.run(_run(case["query"]))
        expected = set(case["expected_sources"])
        got = set(sources)
        ok = bool(got & expected)
        hits += int(ok)
        total += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] {case['query']!r}: expected {expected}, got {got}")

    print(f"\nRouting (graph): {hits}/{total} cases passed")


def _eval_routing_react(cases, services) -> None:
    print("React routing eval: not implemented (use --mode graph)")


# ── SQL eval ──────────────────────────────────────────────────────────────────

def eval_sql() -> None:
    google_key = os.getenv("GOOGLE_API_KEY", "")
    db_path = ROOT / "data" / "demo.db"
    if not db_path.exists():
        print("Demo DB not found — run: python scripts/seed_demo_db.py")
        return
    if not google_key:
        print("SKIP: SQL eval requires GOOGLE_API_KEY")
        return

    from app.config import settings
    from app.llm import make_llm
    from app.tools.database import Database
    from app.graph.sql.subgraph import build_sql_subgraph

    cases_path = ROOT / "eval" / "sql_cases.jsonl"
    cases = [json.loads(l) for l in open(cases_path) if l.strip()]

    db = Database(str(db_path))
    llm = make_llm(settings)
    subgraph = build_sql_subgraph(db, llm, settings)

    first_try_ok = total_ok = total_attempts = 0
    latencies: list[int] = []

    for case in cases:
        t0 = time.perf_counter()
        result = asyncio.run(subgraph.ainvoke({
            "question": case["question"],
            "step_id": "s1",
            "context": "",
            "workspace_id": "",
            "attempts": 0,
            "signatures": [],
            "excluded_tables": [],
        }))
        elapsed = int((time.perf_counter() - t0) * 1000)
        latencies.append(elapsed)

        ok = result.get("status") in ("ok", "empty")
        attempts = result.get("attempts", 1)
        total_attempts += attempts
        fto = result.get("first_try_ok", False) and ok
        first_try_ok += int(fto)
        total_ok += int(ok)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] [{case['id']}] {case['question']!r} — attempts={attempts}, {elapsed}ms")

    n = len(cases)
    latencies.sort()
    p50 = latencies[n // 2] if latencies else 0
    p95 = latencies[int(n * 0.95)] if latencies else 0
    print(f"\nSQL: {total_ok}/{n} correct, {first_try_ok}/{n} first-try")
    print(f"Avg attempts: {total_attempts/n:.2f}, p50={p50}ms, p95={p95}ms")


# ── Multiturn eval ────────────────────────────────────────────────────────────

def eval_multiturn() -> None:
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        print("SKIP: multiturn eval requires GOOGLE_API_KEY")
        return

    from langgraph.checkpoint.memory import InMemorySaver
    from app.config import settings
    from app.llm import make_llm
    from app.tools import Services
    from app.tools.emails import EmailIndex
    from app.tools.database import Database
    from app.graph.build import build_graph
    from app.graph.state import turn_input
    from app.embeddings import default_embedder
    import json as _json

    emails_path = ROOT / "data" / "emails.json"
    with open(emails_path) as f:
        emails = _json.load(f)
    email_index = EmailIndex(emails, default_embedder)

    db_path = ROOT / "data" / "demo.db"
    database = Database(str(db_path)) if db_path.exists() else None
    services = Services(emails=email_index, database=database)

    llm = make_llm(settings)
    graph = build_graph(services, llm, checkpointer=InMemorySaver(), settings=settings)

    cases_path = ROOT / "eval" / "multiturn_cases.jsonl"
    cases = [json.loads(l) for l in open(cases_path) if l.strip()]

    total_turns = hits = 0
    for case in cases:
        thread_id = f"eval-mt-{case['id']}"
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        turn_hits = 0
        for turn in case["turns"]:
            result = asyncio.run(graph.ainvoke(turn_input(turn["query"]), cfg))
            expected = set(turn.get("expected_sources", []))
            got_sources = set()
            for ev in (result.get("evidence") or []):
                src = ev.source if hasattr(ev, "source") else ev.get("source", "")
                got_sources.add(src)
            ok = not expected or bool(got_sources & expected)
            turn_hits += int(ok)
            total_turns += 1
            hits += int(ok)
            status = "OK" if ok else "MISS"
            print(f"  [{status}] [{case['id']}] {turn['query']!r}: expected {expected}, got {got_sources}")
        print()

    print(f"Multiturn: {hits}/{total_turns} turns correct")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["retrieval", "routing", "sql", "multiturn"])
    parser.add_argument("--mode", dest="agent_mode", default="react",
                        choices=["react", "graph"])
    args = parser.parse_args()

    print(f"\n=== {args.mode} eval ===\n")
    if args.mode == "retrieval":
        eval_retrieval()
    elif args.mode == "routing":
        eval_routing(args.agent_mode)
    elif args.mode == "sql":
        eval_sql()
    elif args.mode == "multiturn":
        eval_multiturn()


if __name__ == "__main__":
    main()
