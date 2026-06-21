"""Offline tests for pool retrieval (embedder + rerank stubbed)."""
import numpy as np

import grounding.retrieve as rt
from grounding.graph import SegmentNode, SessionGraph
from grounding.retrieve import RankedSegment, retrieve


def _graph(n):
    nodes, keys = {}, []
    for i in range(n):
        k = f"s::{i}"
        nodes[k] = SegmentNode(k, "s", str(i), f"text {i}", "unit", "section", "main", None, 0, 1, True)
        keys.append(k)
    return SessionGraph(nodes=nodes, by_source={"s": keys}, xref_edges={},
                        term_index={}, extref_index={}, source_text={"s": "…"})


def test_empty_pool_returns_empty():
    assert retrieve(SessionGraph(), "claim", {}) == []


def test_small_pool_reranks_all(monkeypatch):
    g = _graph(5)
    captured = {}

    def fake_rerank(graph, claim, node_keys, top_k):
        captured["keys"] = list(node_keys)
        return [RankedSegment(node_keys[0], "s", "0", graph.nodes[node_keys[0]].text, 0.9)]

    monkeypatch.setattr(rt, "_rerank_nodes", fake_rerank)
    out = retrieve(g, "claim", {})              # no embeddings → small-pool path
    assert len(captured["keys"]) == 5           # every segment reranked
    assert out[0].node_key == "s::0"


def test_prefilter_keeps_nearest(monkeypatch):
    g = _graph(40)                              # > PREFILTER_K → embedder path
    emb = {f"s::{i}": np.array([0, 1, 0, 0], dtype=np.float32) for i in range(40)}
    emb["s::7"] = np.array([1, 0, 0, 0], dtype=np.float32)   # uniquely aligned with claim
    monkeypatch.setattr(rt, "_embed", lambda texts, task: [[1.0, 0.0, 0.0, 0.0]])

    captured = {}
    monkeypatch.setattr(rt, "_rerank_nodes",
                        lambda graph, claim, node_keys, top_k: captured.update(keys=list(node_keys)) or [])

    retrieve(g, "claim", emb, prefilter_k=10)
    assert "s::7" in captured["keys"]           # nearest survives the shortlist
    assert len(captured["keys"]) <= 10
