"""Offline tests for the self-exclusion safeguard (document-as-graph-node).

A claim extracted from the main document must not be grounded by its own clause,
but the document's OTHER clauses (e.g. a cross-referenced definition) stay
available, and genuine sources are never excluded.
"""
from grounding.graph import SegmentNode, SessionGraph
from grounding.pipeline import _DOC_SOURCE_ID, _self_exclude


def _node(key, source_id, text):
    return SegmentNode(
        node_key=key, source_id=source_id, seg_id=key.split("::")[1], text=text,
        kind="unit", seg_type="section", category="main", title=None,
        span_start=0, span_end=len(text), retrievable=True,
    )


CLAIM = "The Receiving Party must keep Confidential Information secret for five years."


def _graph():
    doc_clause = _node("This document::s2", _DOC_SOURCE_ID,
                       "Section 2. " + CLAIM + " This obligation survives termination.")
    doc_def = _node("This document::s1", _DOC_SOURCE_ID,
                    "Section 1. Confidential Information means non-public business data.")
    ext = _node("NDA-Ext::a", "NDA-Ext", CLAIM)   # a real source that happens to contain the claim
    return SessionGraph(
        nodes={n.node_key: n for n in (doc_clause, doc_def, ext)},
        by_source={_DOC_SOURCE_ID: ["This document::s2", "This document::s1"], "NDA-Ext": ["NDA-Ext::a"]},
        xref_edges={}, term_index={}, extref_index={},
        source_text={_DOC_SOURCE_ID: "…", "NDA-Ext": "…"},
    )


def test_excludes_the_documents_own_clause():
    excl = _self_exclude(_graph(), CLAIM)
    assert "This document::s2" in excl          # the clause containing the claim is excluded


def test_keeps_other_document_sections():
    excl = _self_exclude(_graph(), CLAIM)
    assert "This document::s1" not in excl       # the cross-referenced definition stays available


def test_never_excludes_genuine_sources():
    excl = _self_exclude(_graph(), CLAIM)
    assert "NDA-Ext::a" not in excl              # a real source stating the claim is valid support


def test_whitespace_insensitive_match():
    g = _graph()
    spaced = "The   Receiving Party must keep\nConfidential Information secret for five years."
    assert "This document::s2" in _self_exclude(g, spaced)


def test_empty_claim_excludes_nothing():
    assert _self_exclude(_graph(), "   ") == frozenset()
