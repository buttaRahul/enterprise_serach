from __future__ import annotations
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                chunks.append(para[start: start + size])
                start += max(1, size - overlap)
    return chunks or [text]


class CompanySearchInput(BaseModel):
    query: str = Field(..., description="What to search for on the company website")
    top_k: int = Field(3, description="Number of passages to return")


class CompanySiteIndex:
    """
    Semantic search over company website text passages.

    Passages are loaded at construction time (e.g. from crawled URL chunks).
    Embeddings are computed lazily on first search.
    """

    def __init__(
        self,
        passages_or_urls,
        embedder: Callable[[list[str]], np.ndarray],
        *,
        fetch: Callable[[str], str] | None = None,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
    ):
        """
        Two calling modes:
          - passages_or_urls is list[dict] ({"text", "url", "title"}) — used directly.
          - passages_or_urls is list[str] (URLs) + fetch callable — content is fetched
            and chunked at construction time.
        """
        if fetch is not None:
            passages = []
            for url in passages_or_urls:
                text = fetch(url)
                chunks = _chunk(text, chunk_size, chunk_overlap)
                for chunk in chunks:
                    passages.append({"text": chunk, "url": url, "title": url})
            self._passages = passages
        else:
            self._passages = passages_or_urls
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
