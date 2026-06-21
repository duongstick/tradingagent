"""Answer generators.

Both generators obey the same contract: answer ONLY from retrieved context, and abstain
when the context doesn't support an answer. That grounding rule is the whole point — it is
the generation-edge twin of the repo's audit philosophy ("don't let the system fool you").

* ``ExtractiveGenerator`` — offline, no model. Selects and stitches the most query-relevant
  sentences from the retrieved passages and always cites sources. Deterministic.
* ``OpenAIGenerator`` — prompts a chat model with the retrieved context and a strict
  instruction to ground every claim and to say "I don't know" otherwise. Lazy import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .vector_store import Retrieved

ABSTAIN_MESSAGE = (
    "I don't have enough grounded context to answer that. "
    "Try running the pipeline first, or ask about factors, audit findings, "
    "risk, execution, or backtest results."
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Answer:
    """A grounded answer plus the sources it was built from."""

    text: str
    sources: list[str] = field(default_factory=list)
    abstained: bool = False


def _terms(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


class ExtractiveGenerator:
    """Builds an answer by extracting the passages' most query-relevant sentences."""

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def generate(self, query: str, retrieved: list[Retrieved]) -> Answer:
        if not retrieved:
            return Answer(text=ABSTAIN_MESSAGE, sources=[], abstained=True)

        q_terms = _terms(query)
        scored: list[tuple[float, str, str]] = []
        for r in retrieved:
            src = r.chunk.metadata.get("source", r.chunk.doc_id)
            for sent in _sentences(r.chunk.text):
                s_terms = _terms(sent)
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms) / len(q_terms) if q_terms else 0.0
                # Blend lexical overlap with the passage's retrieval score.
                rank = overlap + 0.25 * r.score
                scored.append((rank, sent, src))

        scored.sort(key=lambda t: -t[0])
        picked: list[str] = []
        used_sources: list[str] = []
        seen: set[str] = set()
        for rank, sent, src in scored:
            if rank <= 0:
                continue
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(sent)
            if src not in used_sources:
                used_sources.append(src)
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            # Retrieval returned passages but none lexically matched — fall back to the
            # single best passage rather than fabricating.
            best = retrieved[0]
            src = best.chunk.metadata.get("source", best.chunk.doc_id)
            return Answer(text=best.chunk.text, sources=[src], abstained=False)

        return Answer(text=" ".join(picked), sources=used_sources, abstained=False)


_SYSTEM_PROMPT = (
    "You are vnquant's research assistant. Answer the user's question using ONLY the "
    "provided context passages. If the context does not contain the answer, reply "
    "exactly: 'I don't know based on the available context.' Never invent numbers, "
    "factor names, or results that are not in the context. Be concise."
)


class OpenAIGenerator:
    """Grounded generation via an OpenAI chat model. Lazy import keeps the dep optional."""

    def __init__(self, model: str = "gpt-4o-mini", client=None) -> None:
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

    def generate(self, query: str, retrieved: list[Retrieved]) -> Answer:  # pragma: no cover - network
        if not retrieved:
            return Answer(text=ABSTAIN_MESSAGE, sources=[], abstained=True)
        sources: list[str] = []
        blocks: list[str] = []
        for i, r in enumerate(retrieved, 1):
            src = r.chunk.metadata.get("source", r.chunk.doc_id)
            if src not in sources:
                sources.append(src)
            blocks.append(f"[{i}] (source: {src})\n{r.chunk.text}")
        context = "\n\n".join(blocks)
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        text = resp.choices[0].message.content or ""
        abstained = "i don't know" in text.lower()
        return Answer(text=text.strip(), sources=[] if abstained else sources, abstained=abstained)


def make_generator(cfg):
    """Construct the generator named by ``cfg.backend``."""
    if cfg.backend == "openai":
        return OpenAIGenerator(model=cfg.openai_chat_model)
    if cfg.backend == "offline":
        return ExtractiveGenerator()
    raise ValueError(f"unknown assistant backend: {cfg.backend!r}")
