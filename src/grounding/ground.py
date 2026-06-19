"""Unified per-claim grounding core.

One path shared by production (pipeline.py) and the eval harnesses
(calibrate.py, run_eval.py) so they cannot diverge again. Retrieval backend is
pluggable; the verdict logic and classifier call are identical across backends.

Retrieval backends (build once per request, reused across claims):
  SemchunkRetriever (default) — chunk each source with semchunk, rerank the
    union of chunks across sources, return the top_k as Candidates.
  IldgsRetriever (opt-in, GROUNDING_BACKEND=ildgs) — the enrich → graph → embed
    → route path, wrapped to emit the same Candidate shape.

classifier_input selects what the verdict classifier reads:
  "source"     — full source text of the seed (auto-chunked; robust to silence,
                 the current production behaviour).
  "candidates" — only the reranked top_k passages (Workstream-1 experiment).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from .chunk import chunk_source
from .classify import classify_scores, verdict_from_scores
from .client import get_client
from .extract import extract_span
from .models import Candidate, ClaimResult, EvidenceItem, RankedPassage, SupportingSpan
from .segment import _nlp

ClassifierInput = Literal["source", "candidates"]


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _best_line(claim: str, passage: str) -> SupportingSpan | None:
    """Pick the single sentence of `passage` most relevant to `claim`, via the
    reranker, as the human-facing supporting quote.

    The answer-extractor is extractive QA tuned for *questions*; on declarative
    NLI-style claims it returns near-zero-score fragments (often a heading), so
    it is unreliable for the displayed line. Sentence-level reranking gives the
    exact operative sentence (for a contradiction, the clause it conflicts with),
    with char offsets back into `passage`.
    """
    sents = [s.text.strip() for s in _nlp()(passage).sents if len(s.text.strip()) > 15]
    if not sents:
        return None
    resp = get_client().rerankings.create(
        model="kanon-2-reranker", query=claim, texts=sents,
    )
    if not resp.results:
        return None
    top = resp.results[0]
    sent = sents[top.index]
    start = passage.find(sent)
    if start < 0:
        start = 0
    return SupportingSpan(text=sent, start=start, end=start + len(sent),
                          score=round(top.score, 4))


# ── Retrievers ────────────────────────────────────────────────────────────────

class Retriever(Protocol):
    def candidates(self, claim: str, *, top_k: int = 3) -> list[Candidate]: ...
    def source_text(self, source_id: str) -> str: ...


class SemchunkRetriever:
    """Chunk each source once; rerank the union of chunks per claim.

    `self_exclude_source_id` marks the main document: a claim is never grounded
    by the chunk that restates it verbatim, nor by any later chunk of that same
    document (mirrors pipeline._self_exclude — a claim may only be grounded by
    content preceding it, or by other sources).
    """

    def __init__(self, sources, *, self_exclude_source_id: str | None = None):
        self._text = {s.id: s.text for s in sources}
        self._self_id = self_exclude_source_id
        # (source_id, chunk_index, text) for every chunk, in document order.
        self._chunks: list[tuple[str, int, str]] = []
        for s in sources:
            for i, ch in enumerate(chunk_source(s.text)):
                self._chunks.append((s.id, i, ch))

    def source_text(self, source_id: str) -> str:
        return self._text.get(source_id, "")

    def _eligible(self, claim: str) -> list[tuple[str, int, str]]:
        if not self._self_id:
            return self._chunks
        norm = _norm(claim)
        cutoff: int | None = None
        for sid, idx, ch in self._chunks:
            if sid == self._self_id and norm and norm in _norm(ch):
                cutoff = idx if cutoff is None else min(cutoff, idx)
        if cutoff is None:
            return self._chunks
        return [c for c in self._chunks
                if not (c[0] == self._self_id and c[1] >= cutoff)]

    def candidates(self, claim: str, *, top_k: int = 3) -> list[Candidate]:
        pool = self._eligible(claim)
        if not pool:
            return []
        texts = [c[2] for c in pool]
        resp = get_client().rerankings.create(
            model="kanon-2-reranker", query=claim, texts=texts,
        )
        out: list[Candidate] = []
        for r in resp.results[:top_k]:
            sid, idx, ch = pool[r.index]
            out.append(Candidate(text=ch, source_id=sid, chunk_index=idx,
                                 score=round(r.score, 4)))
        return out


class IldgsRetriever:
    """Adapter over the ILDGS graph path so it emits Candidates like semchunk.

    Built from a prepared SessionGraph + segment embeddings (shared across claims
    by the caller). Cross-reference/term routing is preserved for evidence.
    """

    def __init__(self, graph, seg_embeddings, *, exclude_fn=None):
        self._graph = graph
        self._emb = seg_embeddings
        self._exclude_fn = exclude_fn  # claim -> frozenset[node_key]

    def source_text(self, source_id: str) -> str:
        return self._graph.source_text.get(source_id, "")

    def candidates(self, claim: str, *, top_k: int = 3) -> list[Candidate]:
        from .retrieve import retrieve  # local import: optional backend
        exclude = self._exclude_fn(claim) if self._exclude_fn else frozenset()
        ranked = retrieve(self._graph, claim, self._emb, exclude=exclude,
                          rerank_k=top_k)
        return [Candidate(text=r.text, source_id=r.source_id, chunk_index=0,
                          score=r.score) for r in ranked]


# ── Per-claim grounding ───────────────────────────────────────────────────────

@dataclass
class Grounding:
    candidates: list[Candidate]
    seed: Candidate | None
    relevance: float
    span: SupportingSpan | None
    inextract: float
    answer_score: float
    p_support: float
    p_contra: float
    cls_span: SupportingSpan | None
    verdict: str
    confidence: float
    # Workstream-1 experiment fields (None unless also_reranked=True):
    p_support_rr: float | None = None
    p_contra_rr: float | None = None


def ground(
    claim: str,
    retriever: Retriever,
    *,
    top_k: int = 3,
    classifier_input: ClassifierInput = "source",
    also_reranked: bool = False,
    with_line: bool = False,
) -> Grounding:
    """Retrieve → extract → classify → verdict for one claim. No graph building.

    `with_line=True` replaces the displayed supporting span with the best seed
    sentence (reranked) — a correct human-facing quote. Off by default so the
    eval harness keeps the extractor's answer_score signal unchanged.
    """
    candidates = retriever.candidates(claim, top_k=top_k)
    if not candidates:
        return Grounding([], None, 0.0, None, 1.0, 0.0, 0.0, 0.0, None,
                         "unaddressed", 0.0)

    seed = candidates[0]
    relevance = seed.score
    src_text = retriever.source_text(seed.source_id) or seed.text

    # inextract feeds the verdict (Rule 3 guard); always from the extractor.
    span, inextract = extract_span(claim, [seed.text], relevance_confirmed=True)
    answer_score = span.score if span else 0.0

    # The displayed quote: the extractor span is unreliable on declarative claims,
    # so for the UI use the reranked best sentence of the seed instead.
    if with_line:
        line = _best_line(claim, seed.text)
        if line is not None:
            span = line
            answer_score = line.score

    cand_texts = [c.text for c in candidates]
    cls_texts = [src_text] if classifier_input == "source" else cand_texts
    p_support, p_contra, cls_span = classify_scores(claim, cls_texts)

    p_support_rr = p_contra_rr = None
    if also_reranked:
        # The other signal set, for the offline classifier-input comparison.
        if classifier_input == "source":
            p_support_rr, p_contra_rr, _ = classify_scores(claim, cand_texts)
        else:
            ps2, pc2, _ = classify_scores(claim, [src_text])
            p_support_rr, p_contra_rr = p_support, p_contra
            p_support, p_contra = ps2, pc2  # keep "source" as the primary fields

    verdict, confidence = verdict_from_scores(p_support, p_contra, inextract)
    return Grounding(candidates, seed, relevance, span, inextract, answer_score,
                     p_support, p_contra, cls_span, verdict, confidence,
                     p_support_rr, p_contra_rr)


def to_claim_result(claim, g: Grounding, *, include_passages: bool = False) -> ClaimResult:
    """Build the API ClaimResult from a Grounding (mirrors pipeline rendering)."""
    if g.seed is None:
        return ClaimResult(text=claim.text, range_ref=claim.range_ref,
                           verdict="unaddressed", confidence=0.0)

    if g.verdict == "unaddressed":
        final_span = None
    elif g.verdict == "weak":
        final_span = g.span if g.answer_score > 0.3 else g.cls_span
    else:
        final_span = g.span or g.cls_span

    evidence = [EvidenceItem(source_id=g.seed.source_id, relation="seed",
                             text=g.seed.text, score=g.seed.score)]
    passages: list[RankedPassage] = []
    if include_passages:
        for i, c in enumerate(g.candidates):
            is_seed = i == 0
            passages.append(RankedPassage(
                source_id=c.source_id, score=c.score, passage=c.text,
                line=g.span.text if (is_seed and g.span) else None,
                line_start=g.span.start if (is_seed and g.span) else None,
                line_end=g.span.end if (is_seed and g.span) else None,
            ))

    return ClaimResult(
        text=claim.text, range_ref=claim.range_ref, verdict=g.verdict,
        confidence=g.confidence, span=final_span, source_id=g.seed.source_id,
        evidence=evidence, passages=passages,
    )
