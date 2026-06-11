"""Extract the exact supporting span from a passage using extractive QA.

kanon-answer-extractor has a 512-token native window but chunks internally.
Pass full source texts — do not pre-chunk for this step.
"""
from __future__ import annotations

from .client import get_client
from .models import SupportingSpan


def extract_span(
    claim: str,
    passages: list[str],
    top_k: int = 1,
    relevance_confirmed: bool = False,
) -> tuple[SupportingSpan | None, float]:
    """
    Return (best_span, inextractability_score) for the top passage.

    Pass full source texts; the extractor handles its own chunking.
    Set relevance_confirmed=True when the reranker has already confirmed the passage
    is relevant — this sets ignore_inextractability=True so answers are always returned.

    inextractability_score > 0.5 and > best answer score → no extractable answer.
    Caller uses this to distinguish contradicted vs unaddressed.
    """
    if not passages:
        return None, 1.0

    client = get_client()
    resp = client.extractions.qa.create(
        model="kanon-answer-extractor",
        query=claim,
        texts=passages,
        top_k=top_k,
        ignore_inextractability=relevance_confirmed,
    )

    # extractions are ordered best-first
    best = resp.extractions[0] if resp.extractions else None
    if best is None:
        return None, 1.0

    inextractability = best.inextractability_score
    if not best.answers:
        return None, inextractability

    ans = best.answers[0]
    span = SupportingSpan(
        text=ans.text,
        start=ans.start,
        end=ans.end,
        score=round(ans.score, 4),
    )
    return span, inextractability
