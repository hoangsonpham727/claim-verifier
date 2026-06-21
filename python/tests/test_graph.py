"""Offline tests for the session-graph builder (no API) using fake ILDGS docs."""
from dataclasses import dataclass, field

from grounding.enrich import EnrichedSource
from grounding.graph import (
    build_session_graph,
    crossref_targets,
    extrefs_in,
    retrievable_nodes,
    terms_in,
)


class FSpan:
    def __init__(self, start, end):
        self.start, self.end = start, end

    def decode(self, text):
        return text[self.start:self.end]


@dataclass
class FSeg:
    id: str
    kind: str
    category: str
    level: int
    span: FSpan
    type: str = None
    title: FSpan = None
    children: list = field(default_factory=list)


@dataclass
class FXref:
    start: str
    end: str
    span: FSpan


@dataclass
class FTerm:
    id: str
    name: FSpan
    meaning: FSpan
    mentions: list


@dataclass
class FExt:
    id: str
    name: FSpan
    type: str
    mentions: list
    pinpoints: list = field(default_factory=list)


@dataclass
class FDoc:
    text: str
    segments: list
    crossreferences: list = field(default_factory=list)
    terms: list = field(default_factory=list)
    external_documents: list = field(default_factory=list)


#                0         1         2
#                0123456789012345678901234
TEXT = "AAAA BBBB CCCC DDDD EEEE"  # s1=[0,9) s2=[10,24)


def _doc():
    s1 = FSeg(id="s1", kind="unit", category="main", level=0, span=FSpan(0, 9), type="section")
    s2 = FSeg(id="s2", kind="unit", category="main", level=0, span=FSpan(10, 24), type="section")
    fig = FSeg(id="fig1", kind="figure", category="main", level=1, span=FSpan(10, 14))  # nested in s2
    return FDoc(
        text=TEXT,
        segments=[s1, s2, fig],
        crossreferences=[FXref(start="s1", end="s1", span=FSpan(15, 19))],  # prose in s2 → s1
        terms=[FTerm(id="t1", name=FSpan(0, 4), meaning=FSpan(5, 9), mentions=[FSpan(20, 24)])],
        external_documents=[FExt(id="e1", name=FSpan(20, 24), type="contract", mentions=[FSpan(15, 19)])],
    )


def _graph():
    return build_session_graph([EnrichedSource("src", TEXT, _doc(), ok=True)])


def test_nodes_materialized_with_text_and_offsets():
    g = _graph()
    n = g.nodes["src::s1"]
    assert n.text == "AAAA BBBB"
    assert (n.span_start, n.span_end) == (0, 9)
    assert g.nodes["src::s2"].text == "CCCC DDDD EEEE"


def test_figure_segment_not_retrievable():
    g = _graph()
    keys = set(retrievable_nodes(g))
    assert "src::s1" in keys and "src::s2" in keys
    assert "src::fig1" not in keys


def test_crossref_edge_resolves_issuer_and_target():
    g = _graph()
    # xref span at offset 10 lives in s2; it points to s1
    assert crossref_targets(g, "src::s2") == ["src::s1"]
    assert crossref_targets(g, "src::s1") == []


def test_term_indexed_to_owning_segment():
    g = _graph()
    terms = terms_in(g, "src::s2")          # mention at [15,19) is inside s2
    assert len(terms) == 1
    assert terms[0].name == "AAAA" and terms[0].meaning == "BBBB"


def test_extref_tagged_on_owning_segment():
    g = _graph()
    tags = extrefs_in(g, "src::s2")          # mention at [15,19) is inside s2
    assert len(tags) == 1 and tags[0].ext_type == "contract"


def test_fallback_to_chunks_when_not_ok():
    g = build_session_graph([EnrichedSource("src", "Some fallback text here.", None, ok=False)])
    keys = retrievable_nodes(g)
    assert keys and all(k.startswith("src::chunk:") for k in keys)
    assert g.nodes[keys[0]].text
