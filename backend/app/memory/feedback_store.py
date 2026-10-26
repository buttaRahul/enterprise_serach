"""Feedback store — persists thumbs-up/down with optional SQL correction."""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class FeedbackRecord(BaseModel):
    id: Optional[int] = None
    workspace_id: str
    thread_id: str
    turn_index: int = 0
    question: str
    sql: str = ""
    correction: str = ""
    rating: Literal["up", "down"]
    created_at: float = 0.0


class FeedbackStore:
    """SQLite-backed store for user feedback on SQL answers."""

    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT    NOT NULL,
                    thread_id    TEXT    NOT NULL,
                    turn_index   INTEGER NOT NULL DEFAULT 0,
                    question     TEXT    NOT NULL,
                    sql          TEXT    NOT NULL DEFAULT '',
                    correction   TEXT    NOT NULL DEFAULT '',
                    rating       TEXT    NOT NULL CHECK (rating IN ('up','down')),
                    created_at   REAL    NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_fb_ws ON feedback(workspace_id)")

    def save(self, rec: FeedbackRecord) -> int:
        rec.created_at = time.time()
        with sqlite3.connect(self._path) as con:
            cur = con.execute(
                """INSERT INTO feedback
                   (workspace_id,thread_id,turn_index,question,sql,correction,rating,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (rec.workspace_id, rec.thread_id, rec.turn_index,
                 rec.question, rec.sql, rec.correction, rec.rating, rec.created_at),
            )
            return cur.lastrowid

    def list_positives(self, workspace_id: str, limit: int = 50) -> list[FeedbackRecord]:
        """Return up-rated records for few-shot injection (most recent first)."""
        with sqlite3.connect(self._path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """SELECT * FROM feedback
                   WHERE workspace_id=? AND rating='up' AND sql!=''
                   ORDER BY created_at DESC LIMIT ?""",
                (workspace_id, limit),
            ).fetchall()
        return [FeedbackRecord(**dict(r)) for r in rows]

    def list_corrections(self, workspace_id: str, limit: int = 50) -> list[FeedbackRecord]:
        """Return down-rated records with corrections."""
        with sqlite3.connect(self._path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """SELECT * FROM feedback
                   WHERE workspace_id=? AND rating='down' AND correction!=''
                   ORDER BY created_at DESC LIMIT ?""",
                (workspace_id, limit),
            ).fetchall()
        return [FeedbackRecord(**dict(r)) for r in rows]
