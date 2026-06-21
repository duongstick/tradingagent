"""In-memory vector store and retriever.

A compact cosine-similarity index. Because every embedding is L2-normalized upstream, a
single matrix-vector dot product yields all similarities at once, then we argpartition for
the top-k. No external vector DB needed for a corpus this size; the interface mirrors what
you'd get from FAISS/Chroma so the swap is mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


@dataclass
class Retrieved:
    """A retrieved chunk paired with its similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """Holds chunk embeddings and answers nearest-neighbour queries by cosine similarity."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self._embedder.embed([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[Retrieved]:
        """Return up to ``top_k`` chunks with cosine score >= ``min_score``, best first."""
        if self._matrix is None or len(self._chunks) == 0:
            return []
        q = self._embedder.embed([query])[0]
        sims = self._matrix @ q  # cosine, since both sides are L2-normalized
        k = min(top_k, len(self._chunks))
        # argpartition for the top-k, then sort just those descending.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        out: list[Retrieved] = []
        for i in top_idx:
            score = float(sims[i])
            if score >= min_score:
                out.append(Retrieved(chunk=self._chunks[int(i)], score=score))
        return out
