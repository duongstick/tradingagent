"""Embedding backends.

Two implementations behind one ``Embedder`` protocol:

* ``HashingEmbedder`` — fully offline, deterministic bag-of-n-grams hashed into a fixed
  vector with sub-linear term weighting and L2 normalization. Needs no model download or
  API key, so the showcase runs anywhere; good enough for keyword-ish retrieval over a
  small, factual corpus.
* ``OpenAIEmbedder`` — real dense embeddings via the OpenAI API (lazy import, so the
  dependency is optional). Swap it in by setting ``assistant.backend = "openai"``.

Both return L2-normalized vectors, so a dot product equals cosine similarity downstream.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English function words carry no retrieval signal and, because they appear in
# almost every passage, they inflate cosine similarity for off-topic queries (e.g. "what
# is the best ... in ..."). Dropping them makes similarity reflect content terms, which
# is what lets the abstain guard correctly refuse unrelated questions.
_STOPWORDS = frozenset(
    ["a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "as", "by", "with", "from", "into", "about", "how", "what", "why", "when", "where", "which", "who", "whom", "whose", "do", "does", "did", "done", "can", "could", "should", "would", "may", "might", "will", "shall", "must", "i", "you", "he", "she", "they", "we", "my", "your", "his", "her", "their", "our", "me", "him", "them", "us", "not", "no", "nor", "so", "than", "then", "there", "here", "over", "under", "best", "most", "more", "less", "very", "just", "also", "any", "some", "such", "each", "both", "few", "all", "only", "own", "same", "other"]
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Embedder(Protocol):
    """Maps texts to L2-normalized vectors of a fixed dimension."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Deterministic hashing embedder over unigrams + bigrams.

    Token counts are damped by ``1 + log(count)`` (sub-linear, like TF) so a word
    repeated many times in one chunk doesn't dominate. Signed hashing reduces collision
    bias. Output rows are L2-normalized.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _hash(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if (h[4] & 1) else -1.0
        return bucket, sign

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        toks = _tokenize(text)
        if not toks:
            return vec
        grams = list(toks)
        grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        counts: dict[str, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        for g, c in counts.items():
            bucket, sign = self._hash(g)
            vec[bucket] += sign * (1.0 + math.log(c))
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        mat = np.vstack([self._embed_one(t) for t in texts])
        return _l2_normalize(mat)


class OpenAIEmbedder:
    """Dense embeddings via the OpenAI API. Import is lazy so the dep stays optional."""

    def __init__(self, model: str = "text-embedding-3-small", client=None) -> None:
        self.model = model
        if client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "OpenAI backend requested but 'openai' is not installed. "
                    "Install with: pip install 'vnquant[llm]'"
                ) from exc
            client = OpenAI()
        self._client = client
        self.dim = 1536  # text-embedding-3-small default

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - network
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        resp = self._client.embeddings.create(model=self.model, input=texts)
        mat = np.array([d.embedding for d in resp.data], dtype=np.float64)
        self.dim = mat.shape[1]
        return _l2_normalize(mat)


def make_embedder(cfg) -> Embedder:
    """Construct the embedder named by ``cfg.backend``."""
    if cfg.backend == "openai":
        return OpenAIEmbedder(model=cfg.openai_embed_model)
    if cfg.backend == "offline":
        return HashingEmbedder(dim=cfg.embed_dim)
    raise ValueError(f"unknown assistant backend: {cfg.backend!r}")
