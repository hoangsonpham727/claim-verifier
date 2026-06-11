"""Offline tests for the routing engine (no API; external expansion stubbed)."""
from grounding.graph import Edge, ExtRefTag, SegmentNode, SessionGraph
from grounding.retrieve import RankedSegment
from grounding.route import route


def _node(key, source_id, seg_id, text):
    return SegmentNode(
        node_key=key, source_id=source_id, seg_id=seg_id, text=text,
        kind="unit", seg_type="section", category="main", title=None,
        span_start=0, span_end=len(text), retrievable=True,
    )


def _graph(with_extref=True):
    A = _node("src1::a", "src1", "a", "Alpha clause about confidentiality obligations.")
    B = _node("src1::b", "src1", "b", "Beta definitions section referenced by Alpha.")
    C = _node("src2::c", "src2", "c", "Gamma master agreement clause from another file.")
    g = SessionGraph(
        nodes={A.node_key: A, B.node_key: B, C.node_key: C},
        by_source={"src1": ["src1::a", "src1::b"], "src2": ["src2::c"]},
        xref_edges={"src1::a": [Edge("src1::a", "src1::b", "crossref")]},
        term_index={},
        extref_index={"src1::a": [ExtRefTag("Master Agreement", "contract", 0, 5)]} if with_extref else {},
        source_text={"src1": "…", "src2": "…"},
    )
    return g


def _seed(g, key="src1::a", score=0.9):
    n = g.nodes[key]
    return RankedSegment(n.node_key, n.source_id, n.seg_id, n.text, score)


def test_seed_crossref_and_external_gathered():
    g = _graph()
    calls = []

    def stub_expand(graph, claim, node_key):
        calls.append(node_key)
        c = graph.nodes["src2::c"]
        return RankedSegment(c.node_key, c.source_id, c.seg_id, c.text, 0.5)

    ev = route(g, [_seed(g)], "claim", expand_fn=stub_expand)
    relations = [e.relation for e in ev]
    assert relations == ["seed", "crossref", "external"]
    assert calls == ["src1::a"]                      # expansion ran from the tagged seed
    assert ev[2].source_id == "src2"


def test_external_skipped_when_disabled():
    g = _graph()
    ev = route(g, [_seed(g)], "claim", expand_external=False,
               expand_fn=lambda *a: (_ for _ in ()).throw(AssertionError("should not expand")))
    assert [e.relation for e in ev] == ["seed", "crossref"]


def test_no_expand_without_extref_tag():
    g = _graph(with_extref=False)
    called = []
    route(g, [_seed(g)], "claim", expand_fn=lambda *a: called.append(1))
    assert called == []                              # no tag → no cross-source rerank


def test_dedup_when_crossref_target_already_a_seed():
    g = _graph()
    seeds = [_seed(g, "src1::a", 0.9), _seed(g, "src1::b", 0.8)]
    ev = route(g, seeds, "claim", expand_external=False)
    keys = [(e.source_id, e.text) for e in ev]
    assert len(keys) == len(set(keys))               # B not duplicated via the A→B edge


def test_char_budget_admits_only_seed():
    g = _graph()
    ev = route(g, [_seed(g)], "claim", char_budget=10, expand_external=False)
    assert [e.relation for e in ev] == ["seed"]      # crossref text exceeds remaining budget
