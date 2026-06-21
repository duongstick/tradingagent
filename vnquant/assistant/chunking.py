"""Document chunking for the retrieval index.

Splits a document into overlapping, fixed-size word windows. Overlap matters: a fact
that straddles a naive hard boundary would otherwise be unretrievable from either side,
so adjacent chunks share a tail/head of words.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with provenance back to its source document."""

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict


def chunk_text(
    text: str,
    *,
    doc_id: str,
    chunk_size: int = 90,
    overlap: int = 20,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping word windows.

    ``chunk_size`` is measured in words; ``overlap`` words are shared between neighbours.
    A document shorter than one window yields a single chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    metadata = dict(metadata or {})
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{idx}",
                text=" ".join(window),
                metadata=metadata,
            )
        )
        idx += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks
