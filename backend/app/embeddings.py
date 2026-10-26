from __future__ import annotations
from functools import lru_cache
from typing import Callable
import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def default_embedder(texts: list[str]) -> np.ndarray:
    """Encode texts to L2-normalised float32 vectors using SentenceTransformer."""
    model = _load_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


# Type alias
Embedder = Callable[[list[str]], np.ndarray]
