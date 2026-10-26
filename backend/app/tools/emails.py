from __future__ import annotations
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


class EmailSearchInput(BaseModel):
    content_query: Optional[str] = Field(None, description="What the email is about")
    sender: Optional[str] = Field(None, description="Sender address or name fragment")
    sent_on: Optional[str] = Field(None, description="Exact date YYYY-MM-DD")
    sent_after: Optional[str] = Field(None, description="Only emails after this date YYYY-MM-DD (strict)")
    sent_before: Optional[str] = Field(None, description="Only emails before this date YYYY-MM-DD (strict)")
    top_k: int = Field(5, description="Maximum emails to return")


class EmailIndex:
    """Semantic + filter search over a static list of email dicts."""

    def __init__(
        self,
        emails: list[dict],
        embedder: Callable[[list[str]], np.ndarray],
        retrieval_mode: str = "dense",
    ):
        self._emails = emails
        self._embedder = embedder
        self._retrieval_mode = retrieval_mode
        self._embeddings: Optional[np.ndarray] = None
        self._hybrid_index = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_embeddings(self) -> None:
        if self._embeddings is None:
            texts = [e["content"] for e in self._emails]
            self._embeddings = self._embedder(texts)
            if self._retrieval_mode == "hybrid":
                from app.retrieval.hybrid import HybridIndex
                self._hybrid_index = HybridIndex(texts, self._embedder, lambda_mmr=0.7)

    @staticmethod
    def _parse_date(s: str) -> datetime.date:
        return datetime.strptime(s, "%Y-%m-%d").date()

    @staticmethod
    def _email_date(e: dict) -> datetime.date:
        return datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S").date()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, inp: EmailSearchInput) -> dict:
        """Return {"count": int, "emails": [{"id", "sender", "timestamp", "content", "score?"}]}."""
        if not any([inp.content_query, inp.sender, inp.sent_on, inp.sent_after, inp.sent_before]):
            raise ValueError(
                "At least one of content_query, sender, sent_on, sent_after, sent_before must be provided."
            )

        # Step 1: date and sender filters (fast, deterministic)
        pool = list(self._emails)

        if inp.sender:
            lower = inp.sender.strip().lower()
            pool = [e for e in pool if lower in e["sender"].lower()]

        if inp.sent_on:
            target = self._parse_date(inp.sent_on)
            pool = [e for e in pool if self._email_date(e) == target]

        if inp.sent_after:
            target = self._parse_date(inp.sent_after)
            pool = [e for e in pool if self._email_date(e) > target]

        if inp.sent_before:
            target = self._parse_date(inp.sent_before)
            pool = [e for e in pool if self._email_date(e) < target]

        if not pool:
            return {"count": 0, "emails": []}

        # Step 2: semantic ranking when there is a content query
        if inp.content_query:
            self._ensure_embeddings()
            original_ids = {id(e): i for i, e in enumerate(self._emails)}
            pool_idx = np.array([original_ids[id(e)] for e in pool])

            if self._retrieval_mode == "hybrid" and self._hybrid_index is not None:
                # Hybrid: RRF over the full index, then filter to pool
                global_scored = self._hybrid_index.search(inp.content_query, top_k=len(self._emails))
                idx_to_score = {idx: score for idx, score in global_scored}
                pool_scored = [(pool[j], idx_to_score.get(int(pool_idx[j]), 0.0)) for j in range(len(pool))]
                scored = sorted(pool_scored, key=lambda x: -x[1])
            else:
                # Dense only
                pool_embs = self._embeddings[pool_idx]
                q_emb = self._embedder([inp.content_query])
                sims = cosine_similarity(q_emb, pool_embs).flatten()
                scored = sorted(zip(pool, sims), key=lambda x: -x[1])

            results = [
                {**e, "score": float(s)}
                for e, s in scored[: inp.top_k]
            ]
        else:
            results = pool[: inp.top_k]

        return {"count": len(results), "emails": results}

    def search_tool(self, **kwargs) -> str:
        """Adapter for LangChain StructuredTool (returns JSON string)."""
        import json
        inp = EmailSearchInput(**kwargs)
        out = self.search(inp)
        return json.dumps(out, default=str)
