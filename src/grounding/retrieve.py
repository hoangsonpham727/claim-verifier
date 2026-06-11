"""Pool retrieval over graph segments: embedder pre-filter, then reranker.

Embed every segment across all sources once (cached), cosine-shortlist against
the claim, then rerank the shortlist to pick the seed clause(s). For small pools
the embedder is skipped and the reranker scores all segments directly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .client import get_client
from .graph import SessionGraph, retrievable_nodes

PREFILTER_K = 30   # embedder shortlist size before reranking
RERANK_K = 5       # seeds returned
_EMB_BATCH = 96
_EMB_CACHE: dict[str, np.ndarray] = {}   # text hash → L2-normalized vector


@dataclass
class RankedSegment:
    node_key: str
    source_id: str
    seg_id: str
    text: str
    score: float


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


def _embed(texts: list[str], task: str) -> list[Optional[list[float]]]:
    client = get_client()
    resp = client.embeddings.create(model="kanon-2-embedder", texts=texts, task=task)
    out: list[Optional[list[float]]] = [None] * len(texts)
    for e in resp.embeddings:
        if 0 <= e.index < len(out):
            out[e.index] = e.embedding
    return out


def embed_segments(graph: SessionGraph) -> dict[str, np.ndarray]:
    """Embed all retrievable segment texts (cached by text hash)."""
    keys = retrievable_nodes(graph)
    out: dict[str, np.ndarray] = {}
    miss_keys: list[str] = []
    miss_texts: list[str] = []

    for k in keys:
        h = _hash(graph.nodes[k].text)
        cached = _EMB_CACHE.get(h)
        if cached is not None:
            out[k] = cached
        else:
            miss_keys.append(k)
            miss_texts.append(graph.nodes[k].text)

    for i in range(0, len(miss_texts), _EMB_BATCH):
        bk, bt = miss_keys[i:i + _EMB_BATCH], miss_texts[i:i + _EMB_BATCH]
        for k, vec in zip(bk, _embed(bt, "retrieval/document")):
            if vec is None:
                continue
            arr = _normalize(np.asarray(vec, dtype=np.float32))
            _EMB_CACHE[_hash(graph.nodes[k].text)] = arr
            out[k] = arr

    return out


def retrieve(
    graph: SessionGraph,
    claim: str,
    seg_embeddings: dict[str, np.ndarray],
    *,
    exclude: frozenset[str] = frozenset(),
    prefilter_k: int = PREFILTER_K,
    rerank_k: int = RERANK_K,
) -> list[RankedSegment]:
    """Return up to rerank_k seed segments for the claim, highest-first.

    `exclude` drops node keys from candidacy (used to keep a claim from being
    grounded by its own clause in the main document).
    """
    all_keys = [k for k in retrievable_nodes(graph) if k not in exclude]
    if not all_keys:
        return []

    embeddable = [k for k in all_keys if k in seg_embeddings]
    if len(all_keys) <= prefilter_k or not embeddable:
        shortlist = all_keys                       # small pool → rerank everything
    else:
        qv = _embed([claim], "retrieval/query")[0]
        if qv is None:
            shortlist = embeddable[:prefilter_k]
        else:
            q = _normalize(np.asarray(qv, dtype=np.float32))
            sims = np.stack([seg_embeddings[k] for k in embeddable]) @ q
            order = np.argsort(-sims)[:prefilter_k]
            shortlist = [embeddable[i] for i in order]

    return _rerank_nodes(graph, claim, shortlist, rerank_k)


def _rerank_nodes(graph: SessionGraph, claim: str, node_keys: list[str], top_k: int) -> list[RankedSegment]:
    if not node_keys:
        return []
    client = get_client()
    texts = [graph.nodes[k].text for k in node_keys]
    resp = client.rerankings.create(model="kanon-2-reranker", query=claim, texts=texts)
    out: list[RankedSegment] = []
    for r in resp.results[:top_k]:
        k = node_keys[r.index]
        n = graph.nodes[k]
        out.append(RankedSegment(k, n.source_id, n.seg_id, n.text, round(r.score, 4)))
    return out
