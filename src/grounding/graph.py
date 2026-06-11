"""Session-level legal-document graph built from ILDGS enrichments.

Nodes are ILDGS *segments* (clause/section units) across every source; edges are
intra-document cross-references. Defined terms and external-document references
are indexed per node for the routing engine. When a source can't be enriched, it
falls back to plain `chunk_source` pseudo-segments so it still participates.

All structures are request-scoped (never serialized) — plain dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .chunk import chunk_source
from .enrich import EnrichedSource

# Segment categories/kinds that are not useful retrieval candidates.
_SKIP_CATEGORIES = {"front_matter", "back_matter", "annotation"}
_SKIP_KINDS = {"figure"}


@dataclass
class SegmentNode:
    node_key: str            # f"{source_id}::{seg_id}"
    source_id: str
    seg_id: str
    text: str
    kind: str
    seg_type: Optional[str]
    category: str
    title: Optional[str]
    span_start: int
    span_end: int
    retrievable: bool


@dataclass
class Edge:
    src: str                 # node_key
    dst: str                 # node_key
    kind: str = "crossref"


@dataclass
class TermRef:
    name: str
    meaning: str


@dataclass
class ExtRefTag:
    name: str
    ext_type: str
    start: int
    end: int


@dataclass
class SessionGraph:
    nodes: dict[str, SegmentNode] = field(default_factory=dict)
    by_source: dict[str, list[str]] = field(default_factory=dict)
    xref_edges: dict[str, list[Edge]] = field(default_factory=dict)
    term_index: dict[str, list[TermRef]] = field(default_factory=dict)
    extref_index: dict[str, list[ExtRefTag]] = field(default_factory=dict)
    source_text: dict[str, str] = field(default_factory=dict)


# ── lookup helpers ────────────────────────────────────────────────────────────

def retrievable_nodes(graph: SessionGraph) -> list[str]:
    return [k for k, n in graph.nodes.items() if n.retrievable]


def crossref_targets(graph: SessionGraph, node_key: str) -> list[str]:
    return [e.dst for e in graph.xref_edges.get(node_key, [])]


def terms_in(graph: SessionGraph, node_key: str) -> list[TermRef]:
    return graph.term_index.get(node_key, [])


def extrefs_in(graph: SessionGraph, node_key: str) -> list[ExtRefTag]:
    return graph.extref_index.get(node_key, [])


def _owner_at_offset(spans: list[tuple[int, int, str]], offset: int) -> Optional[str]:
    """Return the node_key of the deepest (smallest) segment span containing offset.

    ILDGS spans are laminar (well-nested, non-crossing), so the most specific
    container is the narrowest span covering the offset.
    """
    best: Optional[str] = None
    best_width = None
    for start, end, node_key in spans:
        if start <= offset < end:
            width = end - start
            if best_width is None or width < best_width:
                best_width, best = width, node_key
    return best


# ── build ─────────────────────────────────────────────────────────────────────

def build_session_graph(enriched: list[EnrichedSource]) -> SessionGraph:
    graph = SessionGraph()

    for e in enriched:
        graph.source_text[e.source_id] = e.text
        graph.by_source.setdefault(e.source_id, [])

        if not e.ok or e.document is None:
            _add_fallback_chunks(graph, e)
            continue

        doc = e.document
        text = doc.text
        spans: list[tuple[int, int, str]] = []   # (start, end, node_key) for containment lookup
        seg_node_key: dict[str, str] = {}         # ILDGS seg id → node_key
        seg_by_id = {seg.id: seg for seg in doc.segments}

        for seg in doc.segments:
            node_key = f"{e.source_id}::{seg.id}"
            seg_node_key[seg.id] = node_key
            retrievable = seg.kind not in _SKIP_KINDS and seg.category not in _SKIP_CATEGORIES
            graph.nodes[node_key] = SegmentNode(
                node_key=node_key,
                source_id=e.source_id,
                seg_id=seg.id,
                text=seg.span.decode(text),
                kind=seg.kind,
                seg_type=seg.type,
                category=seg.category,
                title=seg.title.decode(text) if seg.title else None,
                span_start=seg.span.start,
                span_end=seg.span.end,
                retrievable=retrievable,
            )
            graph.by_source[e.source_id].append(node_key)
            spans.append((seg.span.start, seg.span.end, node_key))

        _index_crossrefs(graph, doc, e.source_id, spans, seg_by_id, seg_node_key)
        _index_terms(graph, doc, spans)
        _index_extrefs(graph, doc, spans)

    return graph


def _add_fallback_chunks(graph: SessionGraph, e: EnrichedSource) -> None:
    cursor = 0
    for i, ch in enumerate(chunk_source(e.text)):
        node_key = f"{e.source_id}::chunk:{i}"
        start = e.text.find(ch, cursor)
        if start < 0:
            start = 0
        end = start + len(ch)
        cursor = end
        graph.nodes[node_key] = SegmentNode(
            node_key=node_key,
            source_id=e.source_id,
            seg_id=f"chunk:{i}",
            text=ch,
            kind="chunk",
            seg_type=None,
            category="main",
            title=None,
            span_start=start,
            span_end=end,
            retrievable=True,
        )
        graph.by_source[e.source_id].append(node_key)


def _index_crossrefs(graph, doc, source_id, spans, seg_by_id, seg_node_key) -> None:
    for xref in doc.crossreferences:
        issuer = _owner_at_offset(spans, xref.span.start)
        if issuer is None:
            continue
        for tgt_id in _expand_range(doc, seg_by_id, xref.start, xref.end):
            dst = seg_node_key.get(tgt_id)
            if dst and dst != issuer:
                graph.xref_edges.setdefault(issuer, []).append(
                    Edge(src=issuer, dst=dst, kind="crossref")
                )


def _expand_range(doc, seg_by_id, start_id: str, end_id: str) -> list[str]:
    """Segment ids covered by a cross-reference's start..end range (document order)."""
    if start_id == end_id:
        return [start_id]
    start_seg, end_seg = seg_by_id.get(start_id), seg_by_id.get(end_id)
    if not start_seg or not end_seg:
        return [start_id]
    lo, hi = start_seg.span.start, end_seg.span.start
    return [seg.id for seg in doc.segments if lo <= seg.span.start <= hi]


def _index_terms(graph, doc, spans) -> None:
    for term in doc.terms:
        name = term.name.decode(doc.text)
        meaning = term.meaning.decode(doc.text)
        for mention in term.mentions:
            owner = _owner_at_offset(spans, mention.start)
            if owner is None:
                continue
            bucket = graph.term_index.setdefault(owner, [])
            if not any(t.name == name for t in bucket):
                bucket.append(TermRef(name=name, meaning=meaning))


def _index_extrefs(graph, doc, spans) -> None:
    for ext in doc.external_documents:
        name = ext.name.decode(doc.text)
        for span in list(ext.mentions) + list(ext.pinpoints):
            owner = _owner_at_offset(spans, span.start)
            if owner is None:
                continue
            graph.extref_index.setdefault(owner, []).append(
                ExtRefTag(name=name, ext_type=ext.type, start=span.start, end=span.end)
            )
