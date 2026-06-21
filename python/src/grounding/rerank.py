"""Rank a source's chunks against a claim with the Kanon 2 Reranker.

This is the retrieval step: the reranker scores each chunk for relevance to the
claim and returns the top_k as Candidates (highest-first). No separate embedding
pre-filter — the reranker ranks the chunk list directly.
"""
from __future__ import annotations

from .client import get_client
from .models import Candidate


def rerank(claim: str, source_id: str, chunks: list[str], top_k: int = 3) -> list[Candidate]:
    """Rank `chunks` by relevance to `claim`; return top_k Candidates, highest-first."""
    if not chunks:
        return []

    client = get_client()
    resp = client.rerankings.create(
        model="kanon-2-reranker",
        query=claim,
        texts=chunks,
    )

    # results are already sorted highest→lowest; map index back to the source chunk
    ranked = [
        Candidate(
            text=chunks[r.index],
            source_id=source_id,
            chunk_index=r.index,
            score=round(r.score, 4),
        )
        for r in resp.results
    ]
    return ranked[:top_k]
