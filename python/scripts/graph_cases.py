"""Practical scenarios for the document-in-graph feature.

Run:  .venv/bin/python scripts/graph_cases.py

Each case prints the verdict and where the evidence came from, and checks it
against an expectation. Cases 1–2 use NO external sources (grounding against the
main document itself); cases 3–4 add external sources.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grounding.models import Source
from grounding.pipeline import verify_sync


def run(title: str, document: str, sources: list[Source], expect: str, expect_src: str | None):
    resp = verify_sync(document, sources, include_passages=True)
    print(f"\n=== {title} ===")
    if not resp.claims:
        print("  (no cited claim detected)")
        return
    r = resp.claims[0]
    ev_src = {e.source_id for e in r.evidence}
    print(f"  claim:   {r.text[:80]}")
    print(f"  verdict: {r.verdict}  ({r.confidence:.2f})   expected: {expect}")
    print(f"  evidence sources: {sorted(ev_src)}")
    print(f"  evidence: {[(e.relation, e.source_id, (e.label or '')[:24]) for e in r.evidence]}")
    ok_verdict = r.verdict in expect.split("|")
    ok_src = expect_src is None or expect_src in ev_src
    print(f"  → {'PASS' if (ok_verdict and ok_src) else 'CHECK'}")


# ── Case 1: same-document cross-reference resolves ────────────────────────────
# The claim (Section 2) restates the term defined in Section 5 of the SAME doc.
# Self-exclusion drops the claim's own clause; Section 5 should ground it.
run(
    "Case 1 — same-document cross-reference (no external source)",
    document=(
        "Section 1. Parties. This Agreement is between Acme Corp and Beta LLC. "
        "Section 2. The Receiving Party must keep Confidential Information secret "
        "for five years, pursuant to Section 5. "
        "Section 5. Term. Confidential Information shall be held in confidence for a "
        "period of five (5) years from the date of disclosure."
    ),
    sources=[],
    expect="supported|weak",
    expect_src="This document",
)

# ── Case 2: self-grounding is prevented ───────────────────────────────────────
# The only text stating the penalty is the claim's own clause; nothing else
# supports it. Without self-exclusion this would be trivially "supported".
run(
    "Case 2 — self-grounding prevented (claim unsupported elsewhere)",
    document=(
        "Section 9. Penalties. The penalty for any breach of this Agreement is "
        "exactly one million dollars, pursuant to Schedule A."
    ),
    sources=[],
    expect="unaddressed|weak|contradicted",
    expect_src=None,
)

# ── Case 3: document claim grounded by an EXTERNAL source ──────────────────────
run(
    "Case 3 — document + external source bridge",
    document=(
        "Confidential Information must be protected for ten years, as set forth in "
        "the Master Agreement."
    ),
    sources=[Source(id="Master-Agreement", text=(
        "Master Agreement. Article 4. Protection Period. All Confidential Information "
        "shall be protected for a period of ten (10) years from disclosure."
    ))],
    expect="supported|weak",
    expect_src="Master-Agreement",
)

# ── Case 4: control — pure external grounding ─────────────────────────────────
run(
    "Case 4 — control: external source only",
    document=(
        "The Disclosing Party retains all intellectual property rights, pursuant to "
        "the Agreement."
    ),
    sources=[Source(id="NDA-1", text=(
        "Section 7. Intellectual Property. All intellectual property rights remain "
        "with the Disclosing Party and no license is granted."
    ))],
    expect="supported|weak",
    expect_src="NDA-1",
)

print("\ndone.")
