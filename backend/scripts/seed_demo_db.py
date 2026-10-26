"""Create and seed the SQLite demo database with employees, projects, and tickets."""
from __future__ import annotations
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "demo.db")


def seed(path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS employees (
        id          INTEGER PRIMARY KEY,
        first_name  TEXT NOT NULL,
        last_name   TEXT NOT NULL,
        email       TEXT UNIQUE NOT NULL,
        title       TEXT NOT NULL,
        department  TEXT NOT NULL,
        salary      INTEGER NOT NULL,
        hire_date   TEXT NOT NULL,
        manager_id  INTEGER REFERENCES employees(id)
    );

    CREATE TABLE IF NOT EXISTS projects (
        id          INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        description TEXT,
        status      TEXT NOT NULL CHECK (status IN ('active','completed','on_hold')),
        priority    TEXT NOT NULL CHECK (priority IN ('P0','P1','P2','P3')),
        owner_id    INTEGER NOT NULL REFERENCES employees(id),
        start_date  TEXT NOT NULL,
        end_date    TEXT,
        budget      INTEGER
    );

    CREATE TABLE IF NOT EXISTS tickets (
        id            INTEGER PRIMARY KEY,
        title         TEXT NOT NULL,
        status        TEXT NOT NULL CHECK (status IN ('open','in_progress','closed')),
        priority      TEXT NOT NULL CHECK (priority IN ('P0','P1','P2','P3')),
        assigned_to   INTEGER REFERENCES employees(id),
        project_id    INTEGER REFERENCES projects(id),
        created_date  TEXT NOT NULL,
        resolved_date TEXT
    );
    """)

    cur.executemany(
        "INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (1,  "Rahul",   "Sharma",  "rahul@company.com",   "Engineering Manager",      "Engineering", 140000, "2022-03-15", None),
            (2,  "Rajesh",  "Kumar",   "rajesh@company.com",  "Senior Software Engineer", "Engineering", 120000, "2021-07-01", 1),
            (3,  "Ram",     "Prasad",  "ram@company.com",     "VP Engineering",           "Management",  180000, "2019-11-20", None),
            (4,  "Raghu",   "Nair",    "raghu@company.com",   "Software Engineer II",     "Engineering", 105000, "2023-01-10", 1),
            (5,  "Raghav",  "Menon",   "raghav@company.com",  "Software Engineer II",     "Engineering", 105000, "2022-09-05", 1),
            (6,  "Raju",    "Pillai",  "raju@company.com",    "Security Engineer",        "Security",    115000, "2020-06-30", 3),
            (7,  "Ramu",    "Iyer",    "ramu@company.com",    "DevOps Lead",              "DevOps",      125000, "2021-02-14", 3),
            (8,  "Roshan",  "Patel",   "roshan@company.com",  "DevOps Engineer",          "DevOps",      108000, "2022-08-22", 7),
            (9,  "Rohit",   "Singh",   "rohit@company.com",   "Software Engineer II",     "Engineering", 105000, "2023-04-17", 1),
            (10, "Rao",     "Verma",   "rao@company.com",     "Software Engineer I",      "Engineering",  92000, "2024-01-08", 1),
            (11, "Ramesh",  "Reddy",   "ramesh@company.com",  "Research Engineer",        "Research",    118000, "2020-12-01", 3),
            (12, "Priya",   "Nair",    "priya@company.com",   "Product Manager",          "Product",     130000, "2021-05-12", 3),
            (13, "Kavita",  "Menon",   "kavita@company.com",  "UX Designer",              "Design",       98000, "2022-11-07", 3),
            (14, "Ramya",   "Krishnan","ramya@company.com",   "Senior Security Engineer", "Security",    125000, "2019-08-15", 6),
        ],
    )

    cur.executemany(
        "INSERT OR IGNORE INTO projects VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (1, "React Forms Overhaul",
             "Rewrite all client-side form validation to use react-hook-form v7.",
             "active", "P1", 1, "2025-01-15", None, 40000),
            (2, "API Performance Tuning",
             "Reduce p99 latency on the public API from 800 ms to under 200 ms.",
             "active", "P0", 2, "2025-02-01", None, 30000),
            (3, "Design System v2",
             "Migrate all UI components to the new Figma-aligned token system.",
             "active", "P2", 13, "2025-02-10", "2025-03-31", 25000),
            (4, "Backend Refactor Q1",
             "Break the monolith into three domain services: auth, search, notifications.",
             "active", "P1", 5, "2025-02-20", None, 60000),
            (5, "Security Hardening Sprint",
             "Remediate all critical CVEs from the Q4 2024 pen-test report.",
             "completed", "P0", 6, "2025-02-10", "2025-02-25", 15000),
            (6, "Infra Scaling Initiative",
             "Auto-scale Kubernetes pods to handle 10× peak traffic.",
             "active", "P1", 8, "2025-02-19", None, 50000),
            (7, "LLM Knowledge Base",
             "Build RAG pipeline over internal documentation for the AI assistant.",
             "active", "P2", 11, "2025-01-01", None, 35000),
            (8, "Mobile App v3",
             "iOS and Android release with offline-first sync and push notifications.",
             "on_hold", "P2", 12, "2025-03-01", None, 80000),
        ],
    )

    cur.executemany(
        "INSERT OR IGNORE INTO tickets VALUES (?,?,?,?,?,?,?,?)",
        [
            (100, "Validation error on login form when email contains a plus sign",
             "closed", "P1", 1, 1, "2025-02-18", "2025-02-20"),
            (101, "LLM gateway latency spike — p99 > 3 s",
             "in_progress", "P0", 9, 2, "2025-02-19", None),
            (102, "CI pipeline fails on Windows agents intermittently",
             "open", "P2", 7, 6, "2025-02-20", None),
            (103, "Pull request #12 review — performance optimizations",
             "closed", "P1", 2, 2, "2025-02-20", "2025-02-21"),
            (104, "Backend refactor: define service boundaries document",
             "in_progress", "P1", 5, 4, "2025-02-20", None),
            (105, "Security patch for CVE-2025-1234 applied — regression test",
             "closed", "P0", 6, 5, "2025-02-10", "2025-02-20"),
            (106, "Infra ticket: add HPA config for search service",
             "in_progress", "P1", 8, 6, "2025-02-19", None),
            (107, "RAG pipeline: chunk size evaluation on internal docs",
             "open", "P2", 11, 7, "2025-02-21", None),
            (108, "Design token migration — button component",
             "open", "P2", 13, 3, "2025-02-21", None),
            (109, "API response time regression after deploy v2.4.1",
             "in_progress", "P0", 2, 2, "2025-02-21", None),
        ],
    )

    con.commit()
    con.close()
    print(f"Demo database seeded at {path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    seed(path)
