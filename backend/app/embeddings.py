from __future__ import annotations
from functools import lru_cache
from typing import Callable
import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load SentenceTransformer; fall back to TF-IDF on torch/torchvision conflicts."""
    try:
        from sentence_transformers import SentenceTransformer
        return ("st", SentenceTransformer(_MODEL_NAME))
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return ("tfidf", TfidfVectorizer(max_features=512, sublinear_tf=True))


def default_embedder(texts: list[str]) -> np.ndarray:
    """
    Encode texts to L2-normalised float32 vectors.
    Uses SentenceTransformer when available, TF-IDF otherwise.
    """
    kind, model = _load_model()

    if kind == "st":
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    # TF-IDF fallback (no torch dependency)
    if not hasattr(model, "vocabulary_"):
        vecs = model.fit_transform(texts).toarray().astype(np.float32)
    else:
        vecs = model.transform(texts).toarray().astype(np.float32)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


# Type alias
Embedder = Callable[[list[str]], np.ndarray]
