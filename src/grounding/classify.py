"""Classify whether passages support a claim; derive verdict + confidence.

Four-verdict decision tree using two independent classifier calls:
  p_support = P(source establishes claim)
  p_contra  = P(source establishes NOT claim)

These are NOT a probability distribution — the universal classifier runs each
hypothesis independently. p_support + p_contra can be 0.3 + 0.7, 0.8 + 0.1,
or 0.6 + 0.5.  Treat them as independent signals.

Decision tree (sequential — first match wins):
  1. max(p_support, p_contra) < τ_low          → UNADDRESSED (source is silent)
  2. p_contra > τ_con AND p_contra > p_support  → CONTRADICTED
  3. p_support > τ_sup AND inextract < τ_inex   → SUPPORTED
  4. else                                        → WEAK

KEY DESIGN POINT — UNADDRESSED is detected by the CLASSIFIER, not the reranker.
An earlier version gated UNADDRESSED on the reranker relevance score, but that
conflates two distinct cases: (a) the source is genuinely silent, and (b) the
source supports the claim but embedder-based retrieval failed to surface the
right chunk.  Both produce relevance ≈ 0.  The classifier auto-chunks the FULL
source text, so when both p_support and p_contra are low the source is truly
silent — robust to retrieval failure.  The reranker is now used only to pick the
best passage for the extractor, never for the verdict.

Thresholds are calibrated offline in eval/calibrate.py — run one score-caching
pass over ContractNLI dev, then grid-search.  The values below are the result.
"""
from __future__ import annotations

from typing import Literal

from .client import get_client
from .models import SupportingSpan

Verdict = Literal["supported", "contradicted", "unaddressed", "weak"]

# ── Thresholds (calibrated in eval/calibrate.py against ContractNLI dev) ───────
# Set from the grid search over eval/scores_dev.json (1037 pairs), objective:
# maximise macro-F1 subject to false-green ≤ 3%.  Result at these values:
#   F1_supported=0.46  F1_contradicted=0.32  F1_unaddressed=0.63  macro=0.47
#   false-green (predicted supported, truth ≠ supported) = 2.9%
#
# τ_sup is deliberately high (0.85): the classifier's p_support distribution for
# truly-supported vs not-supported overlaps heavily below ~0.6, so a high bar is
# the only way to keep false-green under 3%.  Cost is supported recall (~37%) —
# an accepted trade in a legal tool where a false green is worse than a false amber.
_TAU_LOW  = 0.55  # below this for BOTH scores → source is silent → UNADDRESSED
_TAU_CON  = 0.7   # p_contra threshold for CONTRADICTED
_TAU_SUP  = 0.85  # p_support threshold for SUPPORTED (high — see note above)
_TAU_INEX = 0.9   # max inextractability for SUPPORTED (entity-substitution guard)


def _negate(claim: str) -> str:
    """Frame a claim negation for the universal classifier's p_contra call."""
    return f"It is not the case that: {claim}"


def classify_scores(
    claim: str,
    passages: list[str],
) -> tuple[float, float, SupportingSpan | None]:
    """
    Return the raw signals (p_support, p_contra, best_chunk_span) with NO
    verdict rules applied.

    Separated from classify_verdict so the calibration harness can cache raw
    scores once and grid-search thresholds offline without re-hitting the API.
    """
    if not passages:
        return 0.0, 0.0, None

    client = get_client()

    # p_support: does the source establish the claim as stated?
    resp_sup = client.classifications.universal.create(
        model="kanon-universal-classifier",
        query=claim,
        texts=passages,
    )
    best_sup = resp_sup.classifications[0] if resp_sup.classifications else None
    p_support = best_sup.score if best_sup else 0.0

    # p_contra: does the source establish that the claim is false?
    resp_con = client.classifications.universal.create(
        model="kanon-universal-classifier",
        query=_negate(claim),
        texts=passages,
    )
    best_con = resp_con.classifications[0] if resp_con.classifications else None
    p_contra = best_con.score if best_con else 0.0

    # Span from p_support classifier (highest-scoring chunk)
    span: SupportingSpan | None = None
    if best_sup and best_sup.chunks:
        top = best_sup.chunks[0]
        span = SupportingSpan(
            text=top.text,
            start=top.start,
            end=top.end,
            score=round(top.score, 4),
        )

    return p_support, p_contra, span


def verdict_from_scores(
    p_support: float,
    p_contra: float,
    inextract: float = 1.0,
    *,
    tau_low: float = _TAU_LOW,
    tau_con: float = _TAU_CON,
    tau_sup: float = _TAU_SUP,
    tau_inex: float = _TAU_INEX,
) -> tuple[Verdict, float]:
    """
    Pure function: apply the decision tree to cached scores.  No API calls.

    The calibration grid-search calls this directly with candidate thresholds.
    Confidence formulas deliberately exclude the reranker relevance (it is
    unreliable when retrieval fails — see module docstring).
    """
    # Rule 1 — Source is silent: neither support nor contradiction signal
    if max(p_support, p_contra) < tau_low:
        confidence = round(1.0 - max(p_support, p_contra), 4)
        return "unaddressed", confidence

    # Rule 2 — Contradiction: source actively says the opposite.
    # Guard p_contra > p_support rejects ambiguous passages where both are high.
    if p_contra > tau_con and p_contra > p_support:
        confidence = round(p_contra * (1.0 - p_support), 4)
        return "contradicted", confidence

    # Rule 3 — Support: NLI says yes AND extractor can locate the span.
    # inextract guard catches entity-substitution: same topic, but the specific
    # party/date/amount the claim references isn't extractably present.
    if p_support > tau_sup and inextract < tau_inex:
        confidence = round(p_support * (1.0 - inextract), 4)
        return "supported", confidence

    # Rule 4 — Weak: relevant source, but signal is mixed or below threshold
    confidence = round(max(p_support, p_contra), 4)
    return "weak", confidence


def classify_verdict(
    claim: str,
    passages: list[str],
    inextract: float = 1.0,  # extractor P(no answer exists) — feeds Rule 3 guard
) -> tuple[Verdict, float, SupportingSpan | None]:
    """
    Return (verdict, confidence, best_chunk_span).

    Makes two classifier calls (p_support, p_contra), then applies the decision
    tree via verdict_from_scores.  The reranker relevance is intentionally not a
    verdict input (see module docstring); only inextract from the extractor is.
    """
    p_support, p_contra, span = classify_scores(claim, passages)
    verdict, confidence = verdict_from_scores(p_support, p_contra, inextract)
    return verdict, confidence, span
