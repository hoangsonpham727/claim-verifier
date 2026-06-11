"""Routing engine: traverse graph edges from the reranked seed clauses to gather
a bounded, deduped evidence buffer.

Order: seeds (rerank order) → cross-referenced sections → (optional) term
definitions → one external-source expansion when the seed carries an ext-ref tag.
The buffer drives the human-facing evidence and the extractor's seed text; it is
NOT fed to the classifier (the verdict stays on the full source — see pipeline).
"""
from __future__ import annotations

from typing import Callable, Optional

from .graph import SessionGraph, crossref_targets, extrefs_in, retrievable_nodes, terms_in
from .models import EvidenceItem
from .retrieve import RankedSegment, _rerank_nodes

ExpandFn = Callable[[SessionGraph, str, str], Optional[RankedSegment]]


def _trunc(s: str, n: int = 48) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _label(node, relation: str) -> str:
    base = _trunc(node.title) if node.title else (node.seg_type or "passage")
    if relation == "crossref":
        return f"{base} — cross-referenced"
    if relation == "external":
        return f"{node.source_id} — external"
    return base


def default_external_expand(graph: SessionGraph, claim: str, node_key: str) -> Optional[RankedSegment]:
    """Rerank the OTHER sources' segments for the claim; return the best one."""
    src = graph.nodes[node_key].source_id
    others = [k for k in retrievable_nodes(graph) if graph.nodes[k].source_id != src]
    ranked = _rerank_nodes(graph, claim, others, 1)
    return ranked[0] if ranked else None


def route(
    graph: SessionGraph,
    ranked: list[RankedSegment],
    claim: str,
    *,
    max_depth: int = 1,
    char_budget: int = 6000,
    include_terms: bool = False,
    expand_external: bool = True,
    expand_fn: ExpandFn = default_external_expand,
) -> list[EvidenceItem]:
    buffer: list[EvidenceItem] = []
    seen_nodes: set[str] = set()
    seen_text: set[int] = set()
    seen_terms: set[str] = set()
    budget = char_budget

    def admit(node_key: str, relation: str, score: Optional[float]) -> bool:
        nonlocal budget
        if node_key in seen_nodes:
            return False
        node = graph.nodes.get(node_key)
        if node is None or not node.text:
            return False
        h = hash(node.text)
        if h in seen_text:
            return False
        if buffer and len(node.text) > budget:   # seed is always admitted
            return False
        seen_nodes.add(node_key)
        seen_text.add(h)
        budget -= len(node.text)
        buffer.append(
            EvidenceItem(
                source_id=node.source_id,
                relation=relation,
                label=_label(node, relation),
                text=node.text,
                seg_type=node.seg_type,
                start=node.span_start,
                end=node.span_end,
                score=score,
            )
        )
        return True

    # 1. seeds set the agenda
    frontier: list[tuple[str, int]] = []
    for r in ranked:
        if admit(r.node_key, "seed", r.score):
            frontier.append((r.node_key, 0))

    # 2. bounded BFS expansion
    while frontier and budget > 0:
        node_key, depth = frontier.pop(0)
        if depth >= max_depth:
            continue

        for tgt in crossref_targets(graph, node_key):
            if admit(tgt, "crossref", None):
                frontier.append((tgt, depth + 1))

        if include_terms:
            for term in terms_in(graph, node_key):
                if term.name in seen_terms or len(term.meaning) > budget:
                    continue
                seen_terms.add(term.name)
                budget -= len(term.meaning)
                buffer.append(
                    EvidenceItem(
                        source_id=graph.nodes[node_key].source_id,
                        relation="term",
                        label=f'"{_trunc(term.name, 32)}" (defined)',
                        text=term.meaning,
                    )
                )

        if expand_external and extrefs_in(graph, node_key):
            ext = expand_fn(graph, claim, node_key)
            if ext is not None:
                admit(ext.node_key, "external", ext.score)

    return buffer
