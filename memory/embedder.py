"""
memory/embedder.py — sentence-transformers + FAISS utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    emb = model.encode(texts, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return (emb / norms).astype(np.float32)


def build_faiss_index(texts: list[str], index_path: str) -> None:
    import faiss  # type: ignore
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    emb = embed_texts(texts)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, index_path)
    logger.info("FAISS index saved to %s (%d vectors)", index_path, index.ntotal)


def search_faiss(index_path: str, query_text: str, k: int = 5) -> tuple[list[float], list[int]]:
    import faiss  # type: ignore
    index = faiss.read_index(index_path)
    q = embed_texts([query_text])
    distances, indices = index.search(q, k=min(k, index.ntotal))
    return distances[0].tolist(), indices[0].tolist()
