"""
Generate a deterministic synthetic mailbox for hybrid retrieval evaluation (P5.1).

Usage:
    python scripts/make_synthetic_mailbox.py            # writes data/emails_large.json
    python scripts/make_synthetic_mailbox.py --seed 42  # custom seed
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENDERS = [
    "rahul@company.com", "rajesh@company.com", "ram@company.com",
    "raghu@company.com", "raghav@company.com", "raju@company.com",
    "ramu@company.com", "roshan@company.com", "rohit@company.com",
    "rao@company.com", "ramesh@company.com", "priya@company.com",
    "kavita@company.com",
]

# (topic_key, list of paraphrase templates)
TOPICS = [
    ("pull_request", [
        "Pull Request #{n} raised for {feat}. Please review and approve.",
        "PR #{n} is up for review — covers {feat}. Looking for approvals.",
        "Opened a pull request (#{n}) to address {feat}. Review comments welcome.",
        "PR #{n}: {feat} — please have a look before end of day.",
        "Submitted PR #{n} for the {feat} work. Tagging team for review.",
    ]),
    ("deployment", [
        "Deployment pipeline for {env} failed — {reason}. Ticket #{n} filed.",
        "CI/CD pipeline issue on {env}: {reason}. Raised ticket #{n}.",
        "Deploy to {env} broke due to {reason}. See ticket #{n} for details.",
        "{env} deployment is down. Root cause: {reason}. Ticket #{n} created.",
        "Pipeline failure on {env} environment — {reason}. Tracking via #{n}.",
    ]),
    ("ticket", [
        "Ticket #{n} created for {issue}. Assigned to {person}.",
        "Opened issue #{n}: {issue}. Tagging {person} to investigate.",
        "New ticket #{n} — {issue}. {person} to lead the fix.",
        "Raising #{n} for {issue}. {person} please pick this up.",
        "Issue #{n} logged: {issue}. Owner: {person}.",
    ]),
    ("performance", [
        "p99 latency on {service} spiked to {ms}ms. Investigating cause.",
        "Performance regression detected on {service} — p99 is {ms}ms.",
        "{service} is slow: {ms}ms p99 observed in production. Looking into it.",
        "Alert: {service} response time degraded to {ms}ms at p99.",
        "Latency alert on {service}: {ms}ms p99. Team please review.",
    ]),
    ("security", [
        "Security scan found {vuln} in {component}. Patch is in PR #{n}.",
        "Vulnerability {vuln} detected in {component}. Raised PR #{n} to fix.",
        "CVE {vuln} affects {component}. Fix in progress, PR #{n}.",
        "Security issue: {vuln} in {component}. Hotfix PR #{n} ready for review.",
        "{component} has {vuln} vulnerability. PR #{n} submitted.",
    ]),
    ("llm", [
        "LLM gateway {event} — {detail}. Ticket #{n} open.",
        "AI service {event}: {detail}. Tracking via #{n}.",
        "Issue with LLM platform: {event} — {detail}. See #{n}.",
        "LLM service alert — {event}. Details: {detail}. Ticket #{n}.",
        "Reported {event} on LLM gateway. {detail}. #{n} assigned.",
    ]),
    ("project_update", [
        "Project {proj} update: {milestone} completed on schedule.",
        "{proj} milestone: {milestone} done. On track for delivery.",
        "Update on {proj} — {milestone} finished. Next: {next_step}.",
        "{proj}: {milestone} wrapped up. Moving to {next_step}.",
        "Quick update on {proj}: {milestone} complete. {next_step} starting.",
    ]),
    ("meeting", [
        "Reminder: {meeting} scheduled for {time}. Agenda attached.",
        "{meeting} is at {time} today. Please join on time.",
        "Meeting invite: {meeting} at {time}. Required: all hands.",
        "Don't forget {meeting} at {time}. Zoom link in calendar.",
        "{meeting} — {time}. Please review the pre-read before joining.",
    ]),
    ("onboarding", [
        "Welcome {name} to the {dept} team! Starting {start}.",
        "{name} joins {dept} on {start}. Please give them a warm welcome.",
        "New joiner: {name} in {dept}, effective {start}.",
        "{name} is joining {dept} as of {start}. Buddy: {buddy}.",
        "Onboarding {name} to {dept} — start date {start}. Buddy assigned: {buddy}.",
    ]),
    ("infrastructure", [
        "Infra ticket #{n}: {task}. Estimated effort: {effort}.",
        "Infrastructure task #{n} created — {task}. ETA: {effort}.",
        "Raising #{n} for infra work: {task}. Team: {effort} to complete.",
        "Infra #{n}: {task}. Assigned to DevOps, {effort} estimate.",
        "New infra request #{n} — {task}. Timeline: {effort}.",
    ]),
    ("research", [
        "Sharing research findings on {topic}: {finding}.",
        "Research update — {topic}: {finding}.",
        "{topic} investigation complete. Key finding: {finding}.",
        "Paper review on {topic}: {finding}. Implications for Q2.",
        "Circulating research notes on {topic}. TL;DR: {finding}.",
    ]),
    ("review_feedback", [
        "Code review feedback on #{n}: {feedback}. Please address before merge.",
        "Review comments on PR #{n}: {feedback}.",
        "Reviewed PR #{n}. {feedback}. Two changes requested.",
        "#{n} review done. {feedback}. LGTM after fixes.",
        "Feedback on #{n}: {feedback}. Minor nits inline.",
    ]),
]

FEATS = ["form validation", "API caching", "auth middleware", "LLM streaming", "search ranking",
         "token budgeting", "rate limiting", "CI pipeline fix", "DB migration", "UI theming"]
ENVS = ["staging", "production", "dev", "QA"]
REASONS = ["config mismatch", "OOM error", "missing secret", "port conflict", "DB timeout"]
ISSUES = ["memory leak", "null pointer", "race condition", "data loss risk", "stale cache"]
SERVICES = ["API gateway", "search service", "LLM gateway", "auth service", "notification service"]
VULNS = ["CVE-2025-1234", "CVE-2025-5678", "CVE-2025-9012", "SQL injection", "XSS vector"]
COMPS = ["auth module", "parser", "serializer", "admin panel", "email service"]
LLM_EVENTS = ["timeout spike", "context overflow", "rate limit hit", "model fallback triggered", "token budget exceeded"]
LLM_DETAILS = ["increased p99 by 200%", "responses truncated", "retries exhausted", "fallback to gpt-4o-mini", "daily quota reached"]
PROJECTS = ["React Forms Overhaul", "API Performance Tuning", "LLM Knowledge Base", "Design System v2", "Backend Refactor Q1"]
MILESTONES = ["design review", "alpha release", "code freeze", "UAT sign-off", "integration testing"]
NEXT_STEPS = ["beta release", "load testing", "stakeholder demo", "production deploy", "retrospective"]
MEETINGS = ["sprint planning", "architecture review", "1:1", "all-hands", "retrospective", "design sync"]
TIMES = ["10:00 AM", "2:00 PM", "3:30 PM", "9:00 AM", "4:00 PM"]
NAMES = ["Divya", "Kiran", "Arjun", "Sanjay", "Meera", "Lakshmi", "Vishal"]
DEPTS = ["Engineering", "Product", "DevOps", "Research", "Design"]
TASKS = ["bump nginx to 1.24", "add HPA config", "rotate secrets", "update TLS cert", "resize node pool"]
EFFORTS = ["1 day", "3 hours", "half a sprint", "2 days", "1 week"]
RESEARCH_TOPICS = ["RAG architectures", "vector quantisation", "context window optimisation",
                   "fine-tuning vs prompting", "retrieval-augmented generation eval"]
FINDINGS = [
    "512-token chunks with 10% overlap beat larger chunks on recall",
    "few-shot prompting outperforms zero-shot on our eval set by 12%",
    "BM25+dense RRF beats pure dense by 8 NDCG points",
    "long-context models show hallucination spikes beyond 32k tokens",
    "MMR diversity filter improves user satisfaction score by 15%",
]
FEEDBACK = ["nit: rename variable", "missing error handling in edge case",
            "logic is correct but could simplify", "LGTM with minor style fix",
            "please add a test for this path"]


def _fill(template: str, rng: random.Random, n_counter: list[int]) -> str:
    n_counter[0] += 1
    n = 100 + n_counter[0]
    return template.format(
        n=n,
        feat=rng.choice(FEATS),
        env=rng.choice(ENVS),
        reason=rng.choice(REASONS),
        issue=rng.choice(ISSUES),
        service=rng.choice(SERVICES),
        ms=rng.randint(200, 5000),
        vuln=rng.choice(VULNS),
        component=rng.choice(COMPS),
        event=rng.choice(LLM_EVENTS),
        detail=rng.choice(LLM_DETAILS),
        proj=rng.choice(PROJECTS),
        milestone=rng.choice(MILESTONES),
        next_step=rng.choice(NEXT_STEPS),
        meeting=rng.choice(MEETINGS),
        time=rng.choice(TIMES),
        name=rng.choice(NAMES),
        dept=rng.choice(DEPTS),
        start=(date(2025, 3, 1) + timedelta(days=rng.randint(0, 60))).isoformat(),
        buddy=rng.choice(SENDERS),
        task=rng.choice(TASKS),
        effort=rng.choice(EFFORTS),
        topic=rng.choice(RESEARCH_TOPICS),
        finding=rng.choice(FINDINGS),
        feedback=rng.choice(FEEDBACK),
        person=rng.choice(SENDERS).split("@")[0],
    )


def generate(seed: int = 0, n_emails: int = 200) -> list[dict]:
    rng = random.Random(seed)
    counter = [0]
    emails = []
    base_date = date(2025, 1, 1)

    for i in range(n_emails):
        topic_key, templates = rng.choice(TOPICS)
        template = rng.choice(templates)
        content = _fill(template, rng, counter)
        day_offset = rng.randint(0, 89)
        hour = rng.randint(8, 19)
        minute = rng.randint(0, 59)
        ts = f"{(base_date + timedelta(days=day_offset)).isoformat()} {hour:02d}:{minute:02d}:00"
        # Deterministic ID from content hash
        eid = "e" + hashlib.sha1(content.encode()).hexdigest()[:6]
        emails.append({
            "id": eid,
            "timestamp": ts,
            "sender": rng.choice(SENDERS),
            "content": content,
            "topic": topic_key,
        })
    return emails


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "emails_large.json"))
    args = parser.parse_args()

    # Determinism check
    emails_a = generate(args.seed, args.n)
    emails_b = generate(args.seed, args.n)
    assert [e["id"] for e in emails_a] == [e["id"] for e in emails_b], "Not deterministic!"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(emails_a, f, indent=2)
    print(f"Generated {args.n} emails → {args.out}")

    # Also write labelled retrieval queries
    rng = random.Random(args.seed + 1)
    queries = []
    topic_to_ids: dict[str, list[str]] = {}
    for e in emails_a:
        topic_to_ids.setdefault(e["topic"], []).append(e["id"])

    for topic, ids in topic_to_ids.items():
        sample = rng.sample(ids, min(3, len(ids)))
        topic_label = topic.replace("_", " ")
        queries.append({
            "query": f"Find emails about {topic_label}",
            "expected_ids": sample,
            "expected_min": 1,
        })

    out_queries = os.path.join(ROOT, "eval", "email_retrieval_large.jsonl")
    with open(out_queries, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"Generated {len(queries)} retrieval queries → {out_queries}")


if __name__ == "__main__":
    main()
