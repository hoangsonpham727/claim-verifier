"""Orchestration: verify(document, sources) → VerifyResponse.

Segments the document, keeps the cited claims, then grounds each one through
ground.py (retrieve → extract → classify → verdict), fanning out across claims.
The retrieval backend is built once per request and shared by every claim.

Backend selected by env GROUNDING_BACKEND:
  semchunk (default)  chunk sources, rerank the union of chunks.
  ildgs               enrich → graph → embed → route; resolves cross-references.

Both feed the same verdict logic, so production and eval cannot diverge.
"""
from __future__ import annotations

import asyncio
import os
import sys

from .enrich import enrich_sources
from .ground import (
    ClassifierInput,
    IldgsRetriever,
    Retriever,
    SemchunkRetriever,
    ground,
    to_claim_result,
)
from .graph import build_session_graph, retrievable_nodes, SessionGraph
from .models import Claim, ClaimResult, Source, VerifyResponse, VerifySummary
from .retrieve import PREFILTER_K, RERANK_K, embed_segments
from .segment import segment_claims

# The main document is added as a source under this reserved id so its OWN clauses
# (definitions, cross-referenced sections) are available for grounding. A claim is
# never grounded by its own clause (see _self_exclude / SemchunkRetriever).
_DOC_SOURCE_ID = "This document"

# Which text the verdict classifier reads. "source" (full source, auto-chunked)
# is robust to silence and is the calibrated default; "candidates" feeds only the
# reranked passages (see eval Workstream 1). Override via env for experiments.
_CLASSIFIER_INPUT: ClassifierInput = os.getenv("GROUNDING_CLASSIFIER_INPUT", "source")  # type: ignore[assignment]


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _self_exclude(graph: SessionGraph, claim_text: str) -> frozenset[str]:
    """ILDGS-backend self-exclusion: drop the document segment(s) that contain
    the claim verbatim AND every later document segment. A claim may only be
    grounded by content preceding it (or by other sources)."""
    norm = _norm(claim_text)
    if not norm:
        return frozenset()
    excluded: set[str] = set()
    for k, n in graph.nodes.items():
        if n.source_id == _DOC_SOURCE_ID and norm in _norm(n.text):
            excluded.add(k)
    if excluded:
        cutoff = max(graph.nodes[k].span_end for k in excluded)
        for k, n in graph.nodes.items():
            if n.source_id == _DOC_SOURCE_ID and n.span_start >= cutoff:
                excluded.add(k)
    return frozenset(excluded)


def _build_retriever(document: str, sources: list[Source]) -> Retriever:
    """Build the shared retrieval backend once per request."""
    all_sources = list(sources)
    has_doc = bool(document.strip())
    if has_doc:
        all_sources.append(Source(id=_DOC_SOURCE_ID, text=document))

    if os.getenv("GROUNDING_BACKEND", "semchunk").lower() == "ildgs":
        graph = build_session_graph(enrich_sources(all_sources))
        n_nodes = len(retrievable_nodes(graph))
        fires = not os.getenv("GROUNDING_NO_EMBED") and n_nodes > PREFILTER_K
        seg_embeddings = embed_segments(graph) if fires else {}
        if os.getenv("GROUNDING_EMBED_TRACE"):
            print(f"[embed-trace] nodes={n_nodes} prefilter_k={PREFILTER_K} "
                  f"embedder_fired={fires}", file=sys.stderr)
        return IldgsRetriever(
            graph, seg_embeddings,
            exclude_fn=lambda c: _self_exclude(graph, c),
        )

    return SemchunkRetriever(
        all_sources,
        self_exclude_source_id=_DOC_SOURCE_ID if has_doc else None,
    )


def _process_claim(
    claim: Claim,
    retriever: Retriever,
    include_passages: bool = False,
) -> ClaimResult:
    """Ground one cited claim via the unified core — synchronous (to_thread)."""
    g = ground(claim.text, retriever, top_k=RERANK_K,
               classifier_input=_CLASSIFIER_INPUT, with_line=True)
    return to_claim_result(claim, g, include_passages=include_passages)


async def verify(
    document: str = "",
    sources: list[Source] | None = None,
    *,
    claims: list[str] | None = None,
    include_passages: bool = False,
) -> VerifyResponse:
    """
    Verify claims against the sources plus the document itself.

    Claims come from the explicit ``claims`` list (the add-in's selection) when
    given; otherwise the document is segmented and its cited sentences are used.
    """
    sources = sources or []
    if claims is not None:
        cited = [
            Claim(text=c, has_citation=True, range_ref=str(i))
            for i, c in enumerate(claims)
            if c.strip()
        ]
    else:
        cited = [c for c in segment_claims(document) if c.has_citation]

    if not cited:
        return VerifyResponse(
            claims=[],
            summary=VerifySummary(
                total_cited=0, supported=0, contradicted=0, unaddressed=0, weak=0
            ),
        )

    retriever = _build_retriever(document, sources)

    results: list[ClaimResult] = await asyncio.gather(
        *[
            asyncio.to_thread(_process_claim, claim, retriever, include_passages)
            for claim in cited
        ]
    )

    summary = VerifySummary(
        total_cited=len(results),
        supported=sum(1 for r in results if r.verdict == "supported"),
        contradicted=sum(1 for r in results if r.verdict == "contradicted"),
        unaddressed=sum(1 for r in results if r.verdict == "unaddressed"),
        weak=sum(1 for r in results if r.verdict == "weak"),
    )

    return VerifyResponse(claims=list(results), summary=summary)


def verify_sync(
    document: str = "",
    sources: list[Source] | None = None,
    *,
    claims: list[str] | None = None,
    include_passages: bool = False,
) -> VerifyResponse:
    """Synchronous wrapper around verify() — convenience for scripts and tests."""
    return asyncio.run(
        verify(document, sources, claims=claims, include_passages=include_passages)
    )
