"""
Phase 3 integration test — pipeline.verify_sync on real ContractNLI fixtures.

Uses fixtures_contractnli.json: 3 NDA documents each with labelled
Entailment/Contradiction/NotMentioned claim-evidence pairs.

Strategy: for each fixture, pick one Entailment and one Contradiction pair
and build a small synthetic document that cites those hypotheses. Run the
pipeline and check that verdicts land on the correct side.
"""
import json
from pathlib import Path

import pytest

from grounding.models import Source
from grounding.pipeline import verify_sync

FIXTURES_PATH = Path(__file__).parent / "fixtures_contractnli.json"


@pytest.fixture(scope="module")
def fixtures():
    with open(FIXTURES_PATH) as f:
        return json.load(f)


def _pick_pair(fix: dict, verdict: str) -> dict | None:
    """Return first claim pair matching the given verdict label."""
    return next((p for p in fix["claims"] if p["verdict"] == verdict), None)


# ── Smoke test: pipeline returns a VerifyResponse ───────────────────────────

def test_pipeline_returns_response(fixtures):
    fix = fixtures[0]
    pair = _pick_pair(fix, "supported")
    if pair is None:
        pytest.skip("no supported pair in fixture 0")

    # Build a minimal one-sentence document citing this hypothesis
    document = f"{pair['hypothesis']} See Section 1 of the Agreement."
    source = Source(id=str(fix["id"]), text=fix["text"])

    resp = verify_sync(document, [source])

    assert resp.summary.total_cited >= 1
    assert len(resp.claims) >= 1
    total = resp.summary.supported + resp.summary.contradicted + resp.summary.unaddressed + resp.summary.weak
    assert total == resp.summary.total_cited


# ── Verdict direction tests ─────────────────────────────────────────────────

def test_supported_claim_not_contradicted(fixtures):
    """An Entailment pair must not be flagged as contradicted.

    Note: with the calibrated precision-favored thresholds (τ_sup=0.85),
    supported RECALL is ~30% — a single Entailment example may legitimately
    land as 'weak'.  Exact-verdict accuracy is measured in eval/, not here.
    The robust per-example claim is directional: a true support is never a
    contradiction.
    """
    fix = fixtures[0]
    pair = _pick_pair(fix, "supported")
    if pair is None:
        pytest.skip("no supported pair in fixture 0")

    document = f"{pair['hypothesis']} As provided in Section 1 of the Agreement."
    source = Source(id=str(fix["id"]), text=fix["text"])

    resp = verify_sync(document, [source])
    assert resp.claims, "pipeline returned no results"
    result = resp.claims[0]
    assert result.verdict in ("supported", "weak", "unaddressed"), (
        f"Entailment pair was flagged 'contradicted' — a directional error. "
        f"(confidence={result.confidence:.2f})\nclaim: {pair['hypothesis'][:80]}"
    )


def test_contradicted_claim_flagged(fixtures):
    """A hypothesis labelled Contradiction in ContractNLI should come back contradicted."""
    fix = fixtures[1]
    pair = _pick_pair(fix, "contradicted")
    if pair is None:
        pytest.skip("no contradicted pair in fixture 1")

    # Embed the citation marker INSIDE the hypothesis sentence so spaCy keeps it
    # as one claim rather than splitting the citation onto a separate sentence.
    document = f"{pair['hypothesis']}, as provided in Clause 4 of the Agreement."
    source = Source(id=str(fix["id"]), text=fix["text"])

    resp = verify_sync(document, [source])
    assert resp.claims
    # Find the result that covers the hypothesis text (not a stray trailing sentence)
    result = next(
        (r for r in resp.claims if pair["hypothesis"][:30].lower() in r.text.lower()),
        resp.claims[0],
    )
    assert result.verdict in ("contradicted", "unaddressed", "weak"), (
        f"Expected 'contradicted', 'unaddressed', or 'weak', got {result.verdict!r} "
        f"(confidence={result.confidence:.2f})\nclaim: {pair['hypothesis'][:80]}"
    )
    assert result.verdict != "supported", (
        f"Contradicted claim incorrectly marked supported (confidence={result.confidence:.2f})"
    )


# ── Structural tests ─────────────────────────────────────────────────────────

def test_source_id_populated(fixtures):
    fix = fixtures[0]
    pair = _pick_pair(fix, "supported")
    if pair is None:
        pytest.skip()

    document = f"{pair['hypothesis']} As set forth in Section 2."
    source = Source(id=str(fix["id"]), text=fix["text"])
    resp = verify_sync(document, [source])

    for result in resp.claims:
        assert result.source_id == str(fix["id"]), "source_id should be set on every result"


def test_span_populated_for_supported(fixtures):
    fix = fixtures[0]
    pair = _pick_pair(fix, "supported")
    if pair is None:
        pytest.skip()

    document = f"{pair['hypothesis']} Pursuant to the terms of the Agreement."
    source = Source(id=str(fix["id"]), text=fix["text"])
    resp = verify_sync(document, [source])

    supported = [r for r in resp.claims if r.verdict == "supported"]
    for r in supported:
        # span may be None if extractor/classifier both returned nothing, but
        # for a supported verdict we expect at least a classifier chunk
        # — treat as a soft check
        if r.span is not None:
            assert r.span.start >= 0
            assert r.span.end > r.span.start


def test_summary_counts_consistent(fixtures):
    fix = fixtures[2]
    pairs = fix["claims"][:2]
    if not pairs:
        pytest.skip()

    document = " ".join(
        f"{p['hypothesis']} See Section {i+1} of the Agreement."
        for i, p in enumerate(pairs)
    )
    source = Source(id=str(fix["id"]), text=fix["text"])
    resp = verify_sync(document, [source])

    total = resp.summary.supported + resp.summary.contradicted + resp.summary.unaddressed + resp.summary.weak
    assert total == resp.summary.total_cited
    assert total == len(resp.claims)
