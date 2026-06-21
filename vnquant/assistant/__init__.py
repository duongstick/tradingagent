"""Retrieval-augmented research assistant (RAG over the engine's own output).

A grounded Q&A layer: index the methodology notes and live pipeline run-facts, retrieve
the most relevant passages for a question, and answer ONLY from that context — abstaining
when support is too weak. Runs fully offline by default; an OpenAI backend is optional.
"""

from .chunking import Chunk, chunk_text
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder, make_embedder
from .generator import (
    Answer,
    ExtractiveGenerator,
    OpenAIGenerator,
    make_generator,
)
from .knowledge import Document, build_corpus, build_run_documents
from .rag import ResearchAssistant, build_assistant
from .vector_store import Retrieved, VectorStore

__all__ = [
    "Chunk",
    "chunk_text",
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "make_embedder",
    "Answer",
    "ExtractiveGenerator",
    "OpenAIGenerator",
    "make_generator",
    "Document",
    "build_corpus",
    "build_run_documents",
    "Retrieved",
    "VectorStore",
    "ResearchAssistant",
    "build_assistant",
]
