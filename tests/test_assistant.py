"""Tests for the retrieval-augmented research assistant (offline backend)."""

import numpy as np

from vnquant.assistant.chunking import chunk_text
from vnquant.assistant.embeddings import HashingEmbedder
from vnquant.assistant.generator import ExtractiveGenerator
from vnquant.assistant.knowledge import build_corpus, build_run_documents
from vnquant.assistant.rag import ResearchAssistant, build_assistant
from vnquant.assistant.vector_store import VectorStore
from vnquant.default_config import AssistantConfig, Config
from vnquant.pipeline import run_pipeline

# ---- chunking -------------------------------------------------------------------------

def test_chunk_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, doc_id="d", chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # adjacent chunks share `overlap` words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-10:] == second_words[:10]
    # every word appears somewhere
    seen = set()
    for c in chunks:
        seen.update(c.text.split())
    assert len(seen) == 200


def test_chunk_short_text_single_chunk():
    chunks = chunk_text("only a few words", doc_id="d", chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d#0"


# ---- embeddings -----------------------------------------------------------------------

def test_hashing_embedder_normalized_and_deterministic():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["benjamini hochberg false discovery rate"])
    b = emb.embed(["benjamini hochberg false discovery rate"])
    assert a.shape == (1, 128)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a, b)  # deterministic


def test_hashing_embedder_similar_texts_closer():
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed(
        [
            "ledoit wolf shrinkage of the factor covariance matrix",
            "ledoit wolf shrinkage intensity for the covariance",
            "the smart order router schedules twap vwap pov slices",
        ]
    )
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated


# ---- vector store ---------------------------------------------------------------------

def test_vector_store_retrieves_relevant_chunk():
    emb = HashingEmbedder(dim=256)
    store = VectorStore(emb)
    docs = build_corpus()  # methodology only
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d.text, doc_id=d.doc_id, chunk_size=90, overlap=20,
                                 metadata={"source": d.source}))
    store.add(chunks)
    hits = store.search("how does the square root market impact model work", top_k=3)
    assert hits
    assert "method.execution" in hits[0].chunk.doc_id


# ---- generator grounding & abstention -------------------------------------------------

def test_extractive_generator_abstains_without_context():
    gen = ExtractiveGenerator()
    ans = gen.generate("anything", [])
    assert ans.abstained is True
    assert not ans.sources


# ---- end-to-end assistant -------------------------------------------------------------

def test_assistant_answers_methodology_with_sources():
    a = build_assistant(out=None, cfg=AssistantConfig())
    assert a.n_chunks > 0
    ans = a.ask("What is the Deflated Sharpe Ratio and why use it?")
    assert not ans.abstained
    assert ans.sources
    assert "sharpe" in ans.text.lower()


def test_assistant_abstains_on_off_topic():
    a = build_assistant(out=None, cfg=AssistantConfig())
    ans = a.ask("What is the best pho restaurant in Hanoi?")
    assert ans.abstained is True


def test_assistant_empty_query_abstains():
    a = build_assistant(out=None, cfg=AssistantConfig())
    assert a.ask("   ").abstained is True


def test_run_documents_ground_answers_on_live_pipeline():
    cfg = Config()
    cfg.data.n_symbols = 20
    cfg.data.n_days = 300
    out = run_pipeline(cfg)
    run_docs = build_run_documents(out)
    # one summary + one per factor + audit + risk + backtest
    assert any(d.doc_id == "run.backtest" for d in run_docs)
    assert any(d.doc_id == "run.audit" for d in run_docs)

    a = build_assistant(out=out, cfg=AssistantConfig())
    ans = a.ask("How many factors passed FDR in this run?")
    assert not ans.abstained
    assert ans.sources


def test_unknown_backend_raises():
    import pytest
    cfg = AssistantConfig()
    cfg.backend = "telepathy"
    with pytest.raises(ValueError):
        ResearchAssistant(cfg)
