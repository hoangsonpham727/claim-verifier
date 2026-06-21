"""
Workstream C — sweep contradiction-elicitation queries (ROOT-CAUSE fix).

The verdict's `p_contra` comes from one weak prompt — `_negate()` in classify.py:
"It is not the case that: {claim}" — which the universal classifier answers
topically rather than logically, so contradicted barely separates (F1 ~0.32).

Per IQL (https://docs.isaacus.com/iql) the universal classifier supports
{statements}, NOT/AND/OR, `+` (average). This script re-elicits p_contra under
several candidate queries over ContractNLI, reusing the cached p_support from
scores_<split>.json (same source text) for the gap analysis, and reports which
candidate best separates the contradicted class.

⚠ This makes API calls (≈ len(CANDIDATES) × pairs classifier calls). Hand-off:
    python eval/contra_sweep.py --split dev            # ~3 candidates × 1037
    python eval/contra_sweep.py --split dev --max-docs 10   # quick check
Then update _negate() in classify.py with the winning template and re-cache:
    python eval/calibrate.py cache --split dev

Candidate query templates ({claim} is substituted):
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grounding.client import get_client  # noqa: E402

_LABEL_MAP = {"Entailment": "supported", "Contradiction": "contradicted",
              "NotMentioned": "unaddressed"}

# name → query template. Keep "baseline" first as the current production prompt.
#
# ROUND 1 finding (dev): baseline WON decisively. Meta-framed queries that ask
# the model to score a statement *about the document* ("The document states the
# opposite of: X", "According to the document the following is false: X") all
# collapsed to ~0% recall — the universal-classifier query must be a direct
# HYPOTHESIS to verify against the doc, not a meta-instruction. So round 2 only
# tests well-formed direct negations (still proper hypotheses). Expect small
# deltas vs baseline; the real lever is LLM-generated semantic negation.
CANDIDATES: dict[str, str] = {
    "baseline":   "It is not the case that: {claim}",
    "false_that": "It is false that: {claim}",
    "not_true":   "It is not true that: {claim}",
    "untrue":     "{claim} — this is untrue.",
}


def _classify(query: str, source_text: str) -> float:
    resp = get_client().classifications.universal.create(
        model="kanon-universal-classifier", query=query, texts=[source_text],
    )
    return resp.classifications[0].score if resp.classifications else 0.0


def _load_psupport(split: str) -> dict[tuple[int, str], float]:
    path = Path(__file__).parent / f"scores_{split}.json"
    if not path.exists():
        return {}
    return {(r["doc_id"], r["hyp_id"]): r["p_support"]
            for r in json.loads(path.read_text())}


async def _score_doc(doc, labels, psup) -> list[dict]:
    ann = doc["annotation_sets"][0]["annotations"]
    src = doc["text"]

    async def one(hyp_id):
        claim = labels[hyp_id]["hypothesis"]
        row = {"doc_id": doc["id"], "hyp_id": hyp_id,
               "truth": _LABEL_MAP[ann[hyp_id]["choice"]],
               "p_support": psup.get((doc["id"], hyp_id), 0.0)}
        for name, tmpl in CANDIDATES.items():
            q = tmpl.format(claim=claim)
            row[f"pc_{name}"] = round(await asyncio.to_thread(_classify, q, src), 4)
        return row

    return await asyncio.gather(*[one(h) for h in ann])


def _separation(rows, name):
    """Report contradicted detection for candidate `name` via p_contra and gap."""
    key = f"pc_{name}"
    con = [r for r in rows if r["truth"] == "contradicted"]
    non = [r for r in rows if r["truth"] != "contradicted"]
    print(f"\n── {name}  ({CANDIDATES[name]}) ──")
    for tau in [0.5, 0.6, 0.7, 0.8]:
        rec = sum(1 for r in con if r[key] > tau) / max(len(con), 1)
        tp = sum(1 for r in con if r[key] > tau)
        fp = sum(1 for r in non if r[key] > tau)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"   p_contra>{tau}:  rec={rec:5.1%}  prec={prec:5.1%}  F1={f1:.3f}")
    # gap separation (the discriminator the tree uses)
    g_con = [r[key] - r["p_support"] for r in con]
    g_non = [r[key] - r["p_support"] for r in non]
    if g_con and g_non:
        import statistics as st
        print(f"   gap(pc-ps) median:  contradicted={st.median(g_con):+.3f}  "
              f"other={st.median(g_non):+.3f}")


async def main(split: str, max_docs):
    path = Path(__file__).parent / "contract-nli" / f"{split}.json"
    data = json.loads(path.read_text())
    docs, labels = data["documents"], data["labels"]
    if max_docs:
        docs = docs[:max_docs]
    psup = _load_psupport(split)
    if not psup:
        print(f"⚠ No scores_{split}.json — gap analysis will use p_support=0.")

    n = sum(len(d["annotation_sets"][0]["annotations"]) for d in docs)
    print(f"Sweeping {len(CANDIDATES)} candidates over {len(docs)} docs / {n} pairs "
          f"(≈{len(CANDIDATES)*n} classifier calls)\n")

    rows, t0 = [], time.monotonic()
    for i, d in enumerate(docs):
        rows.extend(await _score_doc(d, labels, psup))
        print(f"[{i+1}/{len(docs)}] doc {d['id']}  ({time.monotonic()-t0:.0f}s)")

    out = Path(__file__).parent / f"scores_{split}_contra.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {out}")

    print("\n" + "=" * 60 + "\nCONTRADICTED separability by candidate\n" + "=" * 60)
    for name in CANDIDATES:
        _separation(rows, name)
    print("\n→ Pick the candidate with the best F1 / widest gap, set it in "
          "classify.py _negate(), then re-cache.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.split, args.max_docs))
