"""
Orchestration over the session-level document graph.

verify(document, sources) → VerifyResponse

Per request:
  enrich sources → build graph → embed segments (once, shared)
Per cited claim (fanned out via asyncio.to_thread):
  retrieve seed segments → route (gather evidence) → extract span → classify → ClaimResult

The verdict is produced by the classifier on the FULL seed-source text (unchanged
calibrated thresholds in classify.py). Routing enriches the human-facing evidence
and supplies the extractor's seed text — it does not feed the classifier.
"""
from __future__ import annotations

import asyncio

from .classify import classify_verdict
from .enrich import enrich_sources
from .extract import extract_span
from .graph import build_session_graph, retrievable_nodes, SessionGraph
from .models import (
    Claim,
    ClaimResult,
    EvidenceItem,
    RankedPassage,
    Source,
    VerifyResponse,
    VerifySummary,
)
from .retrieve import PREFILTER_K, embed_segments, retrieve
from .route import route
from .segment import segment_claims

# The main document is enriched into the graph under this reserved source id so
# its OWN clauses (definitions, cross-referenced sections) are available for
# grounding. A claim's own clause is excluded from its evidence (see _self_exclude).
_DOC_SOURCE_ID = "This document"


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _self_exclude(graph: SessionGraph, claim_text: str) -> frozenset[str]:
    """Node keys to exclude for this claim: the document segment(s) that contain
    the claim verbatim — so a claim is never grounded by its own copy. Other
    sources are never excluded (a source that genuinely states the claim IS
    valid support)."""
    norm = _norm(claim_text)
    if not norm:
        return frozenset()
    return frozenset(
        k for k, n in graph.nodes.items()
        if n.source_id == _DOC_SOURCE_ID and norm in _norm(n.text)
    )


def _process_claim(
    claim: Claim,
    graph: SessionGraph,
    seg_embeddings: dict,
    exclude: frozenset[str] = frozenset(),
    include_passages: bool = False,
) -> ClaimResult:
    """Full pipeline for one cited claim — synchronous, called via to_thread."""
    claim_text = claim.text

    ranked = retrieve(graph, claim_text, seg_embeddings, exclude=exclude)
    if not ranked:
        return ClaimResult(
            text=claim.text,
            range_ref=claim.range_ref,
            verdict="unaddressed",
            confidence=0.0,
        )

    # Gather evidence by following graph edges from the seed(s). The external-source
    # expansion costs a rerank call, so only run it when evidence is requested.
    evidence = route(graph, ranked, claim_text, expand_external=include_passages)
    seed = evidence[0]
    best_source_text = graph.source_text[seed.source_id]

    # Char-precise span from the seed clause (stable offsets into seed.text).
    span, inextract = extract_span(claim_text, [seed.text], relevance_confirmed=True)
    answer_score = span.score if span else 0.0

    # 4-rule verdict on the FULL seed-source text (classify.py thresholds).
    verdict, confidence, cls_span = classify_verdict(
        claim_text, [best_source_text], inextract=inextract
    )

    if verdict == "unaddressed":
        final_span = None
    elif verdict == "weak":
        final_span = span if answer_score > 0.3 else cls_span
    else:
        final_span = span or cls_span

    # Deprecated `passages` shim: down-project evidence so the current add-in keeps
    # rendering. The seed carries the exact extracted line; runners-up are passage-only.
    passages: list[RankedPassage] = []
    if include_passages:
        for ev in evidence:
            is_seed = ev.relation == "seed" and ev.text == seed.text
            passages.append(
                RankedPassage(
                    source_id=ev.source_id,
                    score=ev.score or 0.0,
                    passage=ev.text,
                    line=span.text if (is_seed and span) else None,
                    line_start=span.start if (is_seed and span) else None,
                    line_end=span.end if (is_seed and span) else None,
                )
            )

    return ClaimResult(
        text=claim.text,
        range_ref=claim.range_ref,
        verdict=verdict,
        confidence=confidence,
        span=final_span,
        source_id=seed.source_id,
        evidence=evidence,
        passages=passages,
    )


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
    The document is always enriched into the graph so same-document
    cross-references resolve; a claim is never grounded by its own clause.
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

    # The graph spans the SOURCES plus the main document itself (so same-document
    # cross-references resolve). Built/embedded once, shared across claim threads.
    graph_sources = list(sources)
    if document.strip():
        graph_sources.append(Source(id=_DOC_SOURCE_ID, text=document))
    graph = build_session_graph(enrich_sources(graph_sources))
    seg_embeddings = (
        embed_segments(graph) if len(retrievable_nodes(graph)) > PREFILTER_K else {}
    )

    results: list[ClaimResult] = await asyncio.gather(
        *[
            asyncio.to_thread(
                _process_claim, claim, graph, seg_embeddings,
                _self_exclude(graph, claim.text), include_passages,
            )
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
