"""Integration + unit tests for the verdict logic.

Unit tests exercise verdict_from_scores (pure, no API).
Integration tests hit the live Isaacus API via classify_verdict.
"""
import pytest
from grounding.classify import (
    classify_verdict,
    verdict_from_scores,
    _TAU_LOW,
    _TAU_CON,
    _TAU_SUP,
    _TAU_INEX,
)

CLAIM = "The Receiving Party shall not disclose Confidential Information to third parties."
SUPPORTING = (
    "The Receiving Party shall not disclose any Confidential Information to any third party "
    "without the prior written consent of the Disclosing Party. Any disclosure to employees "
    "shall be limited to those with a strict need to know."
)
CONTRADICTING = (
    "The Receiving Party may freely share any Confidential Information with third parties "
    "provided that it gives reasonable notice to the Disclosing Party."
)


# ── Unit tests: pure decision tree (no API) ──────────────────────────────────

def test_verdict_unaddressed_when_both_low():
    # Both signals below τ_low → source is silent
    verdict, conf = verdict_from_scores(p_support=0.1, p_contra=0.1, inextract=1.0)
    assert verdict == "unaddressed"


def test_verdict_supported_when_strong_support_and_extractable():
    verdict, conf = verdict_from_scores(
        p_support=0.9, p_contra=0.1, inextract=_TAU_INEX - 0.1
    )
    assert verdict == "supported"
    assert conf > 0.0


def test_verdict_contradicted_when_contra_dominates():
    verdict, conf = verdict_from_scores(
        p_support=0.2, p_contra=_TAU_CON + 0.1, inextract=0.5
    )
    assert verdict == "contradicted"


def test_verdict_weak_when_support_high_but_inextractable():
    # High p_support but extractor can't locate span → blocked from SUPPORTED
    verdict, conf = verdict_from_scores(
        p_support=0.9, p_contra=0.1, inextract=min(_TAU_INEX + 0.1, 1.0)
    )
    assert verdict == "weak"


def test_verdict_weak_when_both_mid_and_ambiguous():
    # Both above τ_low but neither clears its rule → WEAK
    verdict, conf = verdict_from_scores(
        p_support=0.55, p_contra=0.55, inextract=0.9
    )
    assert verdict == "weak"


# ── Integration tests: live API via classify_verdict ─────────────────────────

def test_classify_supported_passage():
    verdict, confidence, span = classify_verdict(
        CLAIM, [SUPPORTING], inextract=_TAU_INEX - 0.2
    )
    assert verdict == "supported", (
        f"Expected 'supported', got {verdict!r} (confidence={confidence:.2f})"
    )


def test_classify_contradicted_not_supported():
    """A directly-opposing passage must never be flagged 'supported'.

    We assert only the directional guarantee, not the exact verdict: with the
    calibrated τ_low=0.55 and the weak negation-prompt p_contra signal (see
    'diagnose' in eval/calibrate.py), a genuine contradiction whose p_contra
    sits below τ_low legitimately lands as 'unaddressed' or 'weak'.  Catching
    it as 'contradicted' is best-effort, not guaranteed — but greening it would
    be a trust-breaking error, so that is what we test.
    """
    verdict, confidence, span = classify_verdict(
        CLAIM, [CONTRADICTING], inextract=0.5
    )
    assert verdict != "supported", (
        f"Opposing passage was flagged 'supported' — a false green. "
        f"(confidence={confidence:.2f})"
    )


def test_classify_empty_passages():
    verdict, confidence, span = classify_verdict(CLAIM, [])
    assert verdict == "unaddressed"
    assert span is None
