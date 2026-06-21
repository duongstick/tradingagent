"""ResearchAssistant — the RAG orchestrator.

Wires the pieces into one retrieval-augmented pipeline:

    documents -> chunk -> embed -> vector store           (index, once)
    query -> embed -> retrieve top-k -> ground or abstain -> generate   (per question)

The hallucination guard sits between retrieval and generation: if the best retrieved
passage scores below ``abstain_threshold``, the assistant refuses to answer rather than
fabricating. This mirrors the audit framework's job on the data side — both exist to stop
the system from fooling its user.
"""

from __future__ import annotations

from .chunking import chunk_text
from .embeddings import _tokenize, make_embedder
from .generator import ABSTAIN_MESSAGE, Answer, make_generator
from .knowledge import Document, build_corpus
from .vector_store import VectorStore


class ResearchAssistant:
    """A grounded Q&A assistant over the engine's methodology and run facts."""

    def __init__(self, cfg=None) -> None:
        from ..default_config import AssistantConfig  # noqa: PLC0415

        self.cfg = cfg or AssistantConfig()
        self._embedder = make_embedder(self.cfg)
        self._generator = make_generator(self.cfg)
        self._store = VectorStore(self._embedder)
        self._indexed = 0

    def index_documents(self, documents: list[Document]) -> int:
        """Chunk and embed documents into the store. Returns the chunk count added."""
        chunks = []
        for doc in documents:
            chunks.extend(
                chunk_text(
                    doc.text,
                    doc_id=doc.doc_id,
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.chunk_overlap,
                    metadata={"source": doc.source},
                )
            )
        self._store.add(chunks)
        self._indexed += len(chunks)
        return len(chunks)

    def index_corpus(self, out=None) -> int:
        """Convenience: index the methodology corpus plus optional live run facts."""
        return self.index_documents(build_corpus(out))

    @property
    def n_chunks(self) -> int:
        return len(self._store)

    def ask(self, query: str) -> Answer:
        """Answer a question from indexed context, or abstain if support is too weak."""
        if not query or not query.strip():
            return Answer(text=ABSTAIN_MESSAGE, sources=[], abstained=True)

        retrieved = self._store.search(
            query, top_k=self.cfg.top_k, min_score=self.cfg.min_score
        )
        # Hallucination guard, two gates that must BOTH pass:
        #   1. semantic — the best passage clears the cosine floor;
        #   2. lexical  — the best passage shares enough content terms with the query.
        # Hashing-cosine alone is noisy (collisions let off-topic queries score
        # surprisingly high), so the lexical gate is what reliably rejects unrelated
        # questions. If either fails, abstain rather than fabricate.
        if not retrieved or retrieved[0].score < self.cfg.abstain_threshold:
            return Answer(text=ABSTAIN_MESSAGE, sources=[], abstained=True)

        q_terms = set(_tokenize(query))
        best_terms = set(_tokenize(retrieved[0].chunk.text))
        if len(q_terms & best_terms) < self.cfg.min_term_overlap:
            return Answer(text=ABSTAIN_MESSAGE, sources=[], abstained=True)

        return self._generator.generate(query, retrieved)


def build_assistant(out=None, cfg=None) -> ResearchAssistant:
    """Construct an assistant and index the corpus (and live run facts if given)."""
    assistant = ResearchAssistant(cfg)
    assistant.index_corpus(out)
    return assistant
