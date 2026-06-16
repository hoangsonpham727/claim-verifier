"""Unit tests for _self_exclude: claim's own segment plus everything after it."""
from grounding.graph import SegmentNode, SessionGraph
from grounding.pipeline import _DOC_SOURCE_ID, _self_exclude


def _node(key: str, source_id: str, text: str, start: int, end: int) -> SegmentNode:
    return SegmentNode(
        node_key=key,
        source_id=source_id,
        seg_id=key.split("::", 1)[1],
        text=text,
        kind="clause",
        seg_type=None,
        category="main",
        title=None,
        span_start=start,
        span_end=end,
        retrievable=True,
    )


def _graph_with_doc_and_source() -> SessionGraph:
    graph = SessionGraph()
    # Doc has three segments in order: A (preamble), B (the claim), C (later restatement).
    a = _node(f"{_DOC_SOURCE_ID}::A", _DOC_SOURCE_ID, "Preamble text.", 0, 14)
    b = _node(f"{_DOC_SOURCE_ID}::B", _DOC_SOURCE_ID, "The sky is blue.", 14, 30)
    c = _node(f"{_DOC_SOURCE_ID}::C", _DOC_SOURCE_ID, "As stated, the sky is blue.", 30, 57)
    # An external source that genuinely states the same claim — must NOT be excluded.
    s = _node("ext::1", "ext", "The sky is blue, per NOAA.", 0, 26)
    for n in (a, b, c, s):
        graph.nodes[n.node_key] = n
    graph.source_text[_DOC_SOURCE_ID] = "Preamble text.The sky is blue.As stated, the sky is blue."
    graph.source_text["ext"] = s.text
    return graph


def test_excludes_claim_segment_and_everything_after():
    graph = _graph_with_doc_and_source()
    excluded = _self_exclude(graph, "The sky is blue.")
    assert f"{_DOC_SOURCE_ID}::B" in excluded
    assert f"{_DOC_SOURCE_ID}::C" in excluded
    assert f"{_DOC_SOURCE_ID}::A" not in excluded


def test_external_sources_never_excluded():
    graph = _graph_with_doc_and_source()
    excluded = _self_exclude(graph, "The sky is blue.")
    assert "ext::1" not in excluded


def test_empty_claim_excludes_nothing():
    graph = _graph_with_doc_and_source()
    assert _self_exclude(graph, "   ") == frozenset()


def test_claim_not_in_document_excludes_nothing():
    graph = _graph_with_doc_and_source()
    assert _self_exclude(graph, "Completely unrelated assertion.") == frozenset()
