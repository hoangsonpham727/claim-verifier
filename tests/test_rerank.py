"""Integration test — hits the live Isaacus API."""
from grounding.rerank import rerank


CLAIM = "The defendant breached a duty of care owed to the plaintiff."
SOURCE_ID = "s1"

CHUNKS = [
    "This sentence is completely unrelated.",
    "The occupier owed a common duty of care under the Act.",
    "The defendant failed to maintain the premises safely.",
]


def test_rerank_returns_candidates():
    results = rerank(CLAIM, SOURCE_ID, CHUNKS, top_k=2)
    assert len(results) <= 2
    assert all(0.0 <= c.score <= 1.0 for c in results)
    assert all(c.source_id == SOURCE_ID for c in results)


def test_rerank_sorted_descending():
    results = rerank(CLAIM, SOURCE_ID, CHUNKS, top_k=3)
    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_legal_candidate_scores_higher():
    results = rerank(CLAIM, SOURCE_ID, CHUNKS, top_k=3)
    texts = [r.text for r in results]
    legal_ranks = [i for i, t in enumerate(texts) if "unrelated" not in t]
    unrelated_rank = next((i for i, t in enumerate(texts) if "unrelated" in t), None)
    if unrelated_rank is not None and legal_ranks:
        assert min(legal_ranks) < unrelated_rank, "Legal candidates should rank higher than unrelated"


def test_rerank_empty_input():
    assert rerank(CLAIM, SOURCE_ID, []) == []
