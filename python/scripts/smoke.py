"""
Phase 1 smoke test — call each Isaacus model once and print full response shapes.
Run: .venv/bin/python scripts/smoke.py
Results are recorded in NOTES_api.md as the contract for Phase 2.
"""
import json
import pprint
from grounding.client import get_client

client = get_client()

CLAIM = "The defendant breached its duty of care by failing to maintain the premises in a safe condition."
SOURCE = [
    "The occupier failed to repair the broken staircase despite receiving written notice ",
    "from the plaintiff on 14 March 2023. Under the Occupiers Liability Act 1957 s.2, ", 
    "an occupier owes a common duty of care to all lawful visitors.",
]

print("=" * 60)
print("1. EMBEDDINGS — retrieval/query (claim)")
print("=" * 60)
emb_query = client.embeddings.create(
    model="kanon-2-embedder",
    texts=[CLAIM],
    task="retrieval/query",
)
pprint.pprint(emb_query.model_dump())

print("\n" + "=" * 60)
print("2. EMBEDDINGS — retrieval/document (source chunk)")
print("=" * 60)
emb_doc = client.embeddings.create(
    model="kanon-2-embedder",
    texts=SOURCE,
    task="retrieval/document",
)
pprint.pprint(emb_doc.model_dump())

print("\n" + "=" * 60)
print("3. RERANKINGS")
print("=" * 60)
rerank = client.rerankings.create(
    model="kanon-2-reranker",
    query=CLAIM,
    texts=SOURCE,
)
pprint.pprint(rerank.model_dump())

print("\n" + "=" * 60)
print("4. CLASSIFICATIONS (universal, zero-shot)")
print("=" * 60)
labels = ["supported", "contradicted", "unaddressed"]
classify = client.classifications.universal.create(
    model="kanon-universal-classifier",
    query=CLAIM,
    texts=SOURCE,
)
pprint.pprint(classify.model_dump())

print("\n" + "=" * 60)
print("5. EXTRACTIONS (extractive QA)")
print("=" * 60)
try:
    extract = client.extractions.qa.create(
        model="kanon-answer-extractor",
        query=CLAIM,
        texts=SOURCE,
    )
    pprint.pprint(extract.model_dump())
except AttributeError:
    print("Model 'kanon-answer-extractor' not found. Skipping extraction test.")
print("\nDone. Record response field names in NOTES_api.md.")
