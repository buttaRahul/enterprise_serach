from __future__ import annotations
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


class CompanySearchInput(BaseModel):
    query: str = Field(..., description="What to search for on the company website")
    top_k: int = Field(3, description="Number of passages to return")


class CompanySiteIndex:
    """
    Semantic search over company website text passages.

    Passages are loaded at construction time (e.g. from crawled URL chunks).
    Embeddings are computed lazily on first search.
    """

    def __init__(self, passages: list[dict], embedder: Callable[[list[str]], np.ndarray]):
        """
        passages: list of {"text": str, "url": str, "title": str}
        embedder: function that maps list[str] -> ndarray of shape (n, dim)
        """
        self._passages = passages
        self._embedder = embedder
        self._embeddings: Optional[np.ndarray] = None

    def _ensure_embeddings(self) -> None:
        if self._embeddings is None:
            texts = [p["text"] for p in self._passages]
            self._embeddings = self._embedder(texts)

    def search(self, inp: CompanySearchInput) -> dict:
        """Return {"passages": [{"text", "url", "title", "score"}]}."""
        if not self._passages:
            return {"passages": []}

        self._ensure_embeddings()
        q_emb = self._embedder([inp.query])
        sims = cosine_similarity(q_emb, self._embeddings).flatten()
        top_idx = np.argsort(-sims)[: inp.top_k]
        passages = [
            {**self._passages[i], "score": float(sims[i])}
            for i in top_idx
        ]
        return {"passages": passages}

    def search_tool(self, query: str, top_k: int = 3) -> str:
        """Adapter for LangChain StructuredTool."""
        import json
        result = self.search(CompanySearchInput(query=query, top_k=top_k))
        return json.dumps(result)

    @classmethod
    def from_urls(
        cls,
        urls: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        chunk_size: int = 1000,
    ) -> "CompanySiteIndex":
        """Load and chunk URLs, then build an index. Requires `unstructured`."""
        from langchain_community.document_loaders import UnstructuredURLLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        passages = [
            {"text": c.page_content, "url": c.metadata.get("source", ""), "title": c.metadata.get("title", "")}
            for c in chunks
        ]
        return cls(passages, embedder)
