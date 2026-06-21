"""Unit tests for the unified grounding core (no API calls)."""
from __future__ import annotations

import grounding.ground as G
from grounding.ground import Grounding, SemchunkRetriever, ground, to_claim_result
from grounding.models import Candidate, Claim, Source, SupportingSpan


class FakeRetriever:
    """Stand-in retriever — no reranker/API."""
    def __init__(self, candidates, texts):
        self._c = candidates
        self._t = texts

    def candidates(self, claim, *, top_k=3):
        return self._c[:top_k]

    def source_text(self, source_id):
        return self._t.get(source_id, "")


def _patch_signals(monkeypatch, *, inextract, p_support, p_contra, record=None):
    monkeypatch.setattr(
        G, "extract_span",
        lambda claim, texts, **kw: (SupportingSpan(text="x", start=0, end=1, score=0.5), inextract),
    )

    def fake_classify(claim, texts):
        if record is not None:
            record.append(list(texts))
        return p_support, p_contra, None

    monkeypatch.setattr(G, "classify_scores", fake_classify)


def test_ground_no_candidates_is_unaddressed(monkeypatch):
    g = ground("c", FakeRetriever([], {}))
    assert g.verdict == "unaddressed" and g.seed is None


def test_ground_classifier_input_source_vs_candidates(monkeypatch):
    cand = [Candidate(text="CANDIDATE", source_id="s1", chunk_index=0, score=0.9)]
    retr = FakeRetriever(cand, {"s1": "FULL SOURCE TEXT"})

    seen: list[list[str]] = []
    _patch_signals(monkeypatch, inextract=0.1, p_support=0.95, p_contra=0.1, record=seen)

    g = ground("claim", retr, classifier_input="source")
    assert seen[-1] == ["FULL SOURCE TEXT"]          # classifier read full source
    assert g.verdict == "supported"                  # high p_support, low inextract

    seen.clear()
    g2 = ground("claim", retr, classifier_input="candidates")
    assert seen[-1] == ["CANDIDATE"]                 # classifier read the candidate


def test_ground_also_reranked_records_both_sets(monkeypatch):
    cand = [Candidate(text="CAND", source_id="s1", chunk_index=0, score=0.8)]
    retr = FakeRetriever(cand, {"s1": "SRC"})
    _patch_signals(monkeypatch, inextract=0.2, p_support=0.9, p_contra=0.2)
    g = ground("claim", retr, classifier_input="source", also_reranked=True)
    # primary fields stay full-source; rr fields populated for the A/B.
    assert g.p_support == 0.9
    assert g.p_support_rr is not None and g.p_contra_rr is not None


def test_to_claim_result_unaddressed_has_no_span(monkeypatch):
    g = Grounding(candidates=[], seed=None, relevance=0.0, span=None,
                  inextract=1.0, answer_score=0.0, p_support=0.0, p_contra=0.0,
                  cls_span=None, verdict="unaddressed", confidence=0.1)
    r = to_claim_result(Claim(text="c"), g)
    assert r.verdict == "unaddressed" and r.span is None


def test_semchunk_self_exclusion_drops_verbatim_and_later():
    # The document restates the claim verbatim; that chunk and all later doc
    # chunks must be excluded so the document can't ground itself.
    claim = "The receiving party shall keep information confidential."
    doc = (f"Preamble clause about scope. {claim} A later clause that follows.")
    retr = SemchunkRetriever(
        [Source(id="DOC", text=doc), Source(id="other", text="unrelated text")],
        self_exclude_source_id="DOC",
    )
    eligible = retr._eligible(claim)
    # No eligible DOC chunk may contain the claim verbatim.
    norm = G._norm(claim)
    assert all(not (sid == "DOC" and norm in G._norm(txt))
               for sid, _, txt in eligible)
    # The 'other' source remains eligible.
    assert any(sid == "other" for sid, _, _ in eligible)
