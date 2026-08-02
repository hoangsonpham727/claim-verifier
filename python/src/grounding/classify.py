"""Verdict logic: turn classifier scores into supported/contradicted/unaddressed.

Two independent classifier calls give two signals — they are NOT a probability
distribution and need not sum to 1:
  p_support = P(source establishes the claim)
  p_contra  = P(source establishes the claim's negation)

Sequential decision tree, first match wins:
  1. max(p_support, p_contra) < τ_low           → UNADDRESSED (source is silent)
  2. p_contra > τ_con and p_contra > p_support  → CONTRADICTED
  3. p_support > τ_sup and inextract < τ_inex   → SUPPORTED
  4. otherwise                                   → WEAK

Why UNADDRESSED comes from the classifier, not the reranker: a low reranker
score is ambiguous — the source may be genuinely silent, or it may support the
claim while retrieval simply missed the right chunk. Both look identical. The
classifier reads the FULL source (auto-chunking internally), so two low scores
mean real silence. The reranker only picks the extractor's passage; it never
decides a verdict.

Thresholds below are grid-searched offline by eval/calibrate.py.
"""
from __future__ import annotations

from typing import Literal

from .client import get_client
from .models import SupportingSpan

Verdict = Literal["supported", "contradicted", "unaddressed", "weak"]

# ── Calibrated thresholds ─────────────────────────────────────────────────────
# Grid-searched over eval/scores_dev.json (ContractNLI dev, 1037 pairs) to
# maximise macro-F1 subject to false-green ≤ 3%. Result at these values:
#   F1: supported 0.46 · contradicted 0.32 · unaddressed 0.66 · macro 0.48
#   false-green (predicted supported, truth ≠ supported) = 2.9%
#
# τ_sup is deliberately high: p_support for truly-supported vs not-supported
# overlaps heavily below ~0.6, so a high bar is the only way to hold false-green
# under 3%. It costs recall (~37%) — the right trade for a legal tool, where a
# false "supported" is far worse than a cautious "needs review".
_TAU_LOW  = 0.55  # both scores below this → source is silent → UNADDRESSED
_TAU_CON  = 0.7   # p_contra needed for CONTRADICTED
_TAU_SUP  = 0.85  # p_support needed for SUPPORTED (see note above)
_TAU_INEX = 0.9   # max inextractability for SUPPORTED (entity-substitution guard)

# _TAU_MARGIN — minimum (p_contra − p_support) gap for CONTRADICTED. Held at 0.0,
#   which reduces to the plain "p_contra > p_support" guard: on dev, requiring a
#   gap traded contradicted gains for a net macro-F1 loss because p_contra itself
#   is the weak signal.
# _TAU_UNADDR — inextract above this (with low p_support) ⇒ UNADDRESSED. The
#   extractor finding no answer is strong evidence of silence (median 0.85 for
#   unaddressed vs 0.09 for supported); 1.0 disables the rule, 0.7 lifts
#   F1_unaddressed 0.63 → 0.66 at unchanged false-green.
_TAU_MARGIN = 0.0
_TAU_UNADDR = 0.7


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
    tau_margin: float = _TAU_MARGIN,
    tau_unaddr: float = _TAU_UNADDR,
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

    # Rule 2 — Contradiction: source actively says the opposite. The margin
    # (p_contra − p_support) is the real discriminator (see threshold notes); the
    # absolute τ_con stays as a floor. tau_margin=0.0 reduces this to the old
    # "p_contra > p_support" guard.
    if p_contra > tau_con and (p_contra - p_support) > tau_margin:
        confidence = round(p_contra * (1.0 - p_support), 4)
        return "contradicted", confidence

    # Rule 3 — Support: NLI says yes AND extractor can locate the span.
    # inextract guard catches entity-substitution: same topic, but the specific
    # party/date/amount the claim references isn't extractably present.
    if p_support > tau_sup and inextract < tau_inex:
        confidence = round(p_support * (1.0 - inextract), 4)
        return "supported", confidence

    # Rule 3.5 — Silence by inextractability: the extractor found no answer and
    # support is weak ⇒ the source does not address the claim. Reclaims pairs
    # that would otherwise fall to WEAK. tau_unaddr=1.0 disables this rule.
    if inextract > tau_unaddr and p_support < tau_sup:
        confidence = round(inextract * (1.0 - p_support), 4)
        return "unaddressed", confidence

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
