"""Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion (P5.2)."""
from __future__ import annotations
from typing import Callable, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class HybridIndex:
    """
    Dense + BM25 (Okapi) search combined via RRF (k=60).

    Parameters
    ----------
    texts      : list of text strings to index
    embedder   : function that maps list[str] → (n, dim) float32 array
    lambda_mmr : if > 0, apply Maximal Marginal Relevance diversity filter (λ=lambda_mmr)
    """

    RRF_K = 60

    def __init__(
        self,
        texts: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        lambda_mmr: float = 0.0,
    ):
        self._texts = texts
        self._embedder = embedder
        self._lambda_mmr = lambda_mmr
        self._dense_vecs: Optional[np.ndarray] = None
        self._bm25 = None

    # ── Lazy init ──────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if self._dense_vecs is not None:
            return
        self._dense_vecs = self._embedder(self._texts)
        from rank_bm25 import BM25Okapi
        tokenised = [t.lower().split() for t in self._texts]
        self._bm25 = BM25Okapi(tokenised)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Return (index, rrf_score) pairs, sorted by descending RRF score.
        """
        self._ensure_index()
        n = len(self._texts)

        # ── Dense rank ────────────────────────────────────────────────────────
        q_vec = self._embedder([query])
        dense_sims = cosine_similarity(q_vec, self._dense_vecs).flatten()
        dense_order = np.argsort(-dense_sims)  # descending

        # ── BM25 rank ─────────────────────────────────────────────────────────
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_order = np.argsort(-bm25_scores)  # descending

        # ── RRF fusion ────────────────────────────────────────────────────────
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_order)}
        bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_order)}

        rrf: dict[int, float] = {}
        for i in range(n):
            dr = dense_rank.get(i, n)
            br = bm25_rank.get(i, n)
            rrf[i] = 1.0 / (self.RRF_K + dr) + 1.0 / (self.RRF_K + br)

        scored = sorted(rrf.items(), key=lambda x: -x[1])

        if self._lambda_mmr > 0:
            scored = self._mmr(scored, top_k)
        else:
            scored = scored[:top_k]

        return scored

    # ── MMR diversity filter ───────────────────────────────────────────────────

    def _mmr(self, scored: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
        """Maximal Marginal Relevance (λ controls relevance vs diversity)."""
        lam = self._lambda_mmr
        candidates = list(scored)
        selected: list[tuple[int, float]] = []

        while candidates and len(selected) < top_k:
            if not selected:
                best = max(candidates, key=lambda x: x[1])
            else:
                sel_vecs = self._dense_vecs[[s[0] for s in selected]]

                def _mmr_score(c_idx: int, c_score: float) -> float:
                    c_vec = self._dense_vecs[c_idx : c_idx + 1]
                    max_sim = float(cosine_similarity(c_vec, sel_vecs).max())
                    return lam * c_score - (1 - lam) * max_sim

                best = max(candidates, key=lambda x: _mmr_score(x[0], x[1]))
            selected.append(best)
            candidates = [c for c in candidates if c[0] != best[0]]

        return selected
