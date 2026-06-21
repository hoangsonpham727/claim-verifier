"""
Phase 6 eval — run the grounding pipeline over ContractNLI and report
precision / recall / F1 per verdict class.

Usage:
    # Dev set (61 docs, 1037 pairs) — recommended first run
    python eval/run_eval.py --split dev

    # Limit to N documents for a quick smoke-check
    python eval/run_eval.py --split dev --max-docs 10

    # Full train set (423 docs) — slow
    python eval/run_eval.py --split train

Label mapping:
    ContractNLI Entailment   → supported
    ContractNLI Contradiction → contradicted  (weak is acceptable)
    ContractNLI NotMentioned  → unaddressed   (weak is acceptable)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grounding.models import Source
from grounding.pipeline import verify

# ── ContractNLI label → our verdict ──────────────────────────────────────────

_LABEL_MAP = {
    "Entailment":    "supported",
    "Contradiction": "contradicted",
    "NotMentioned":  "unaddressed",
}

# For evaluation: WEAK is acceptable for Contradiction and NotMentioned,
# but is a miss for Entailment (we want to commit to "supported").
_ACCEPTABLE = {
    "supported":    {"supported"},
    "contradicted": {"contradicted", "weak"},
    "unaddressed":  {"unaddressed", "weak"},
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_split(split: str) -> tuple[list[dict], dict]:
    path = Path(__file__).parent / "contract-nli" / f"{split}.json"
    with open(path) as f:
        data = json.load(f)
    return data["documents"], data["labels"]


# ── Per-document eval ─────────────────────────────────────────────────────────

async def eval_document(
    doc: dict,
    labels: dict,
) -> list[dict]:
    """
    Run the pipeline over all hypotheses in one NDA document.

    Returns a list of result dicts: {hyp_id, truth, predicted, acceptable}.
    """
    hyp_ids = list(doc["annotation_sets"][0]["annotations"].keys())
    annotations = doc["annotation_sets"][0]["annotations"]

    # Pass the hypotheses as explicit claims against the NDA source. Crucially we
    # do NOT build a synthetic document from the hypotheses (the old approach,
    # which grounded the hypotheses against each other → ~96% false-green). Each
    # claim is verified only against the document's source text.
    hyps = [labels[hyp_id]["hypothesis"] for hyp_id in hyp_ids]
    source = Source(id=str(doc["id"]), text=doc["text"])

    try:
        response = await verify(document="", sources=[source], claims=hyps)
    except Exception as exc:
        print(f"  [ERROR] doc {doc['id']}: {exc}")
        return []

    # verify() builds Claim(text=c) verbatim from `claims`, so match by exact text.
    verdict_by_text: dict[str, str] = {r.text: r.verdict for r in response.claims}

    rows = []
    for hyp_id, hyp_text in zip(hyp_ids, hyps):
        truth_label = annotations[hyp_id]["choice"]
        truth_verdict = _LABEL_MAP[truth_label]
        predicted = verdict_by_text.get(hyp_text, "unmatched")

        acceptable = predicted in _ACCEPTABLE.get(truth_verdict, set())
        rows.append({
            "doc_id":    doc["id"],
            "hyp_id":    hyp_id,
            "truth":     truth_verdict,
            "predicted": predicted,
            "acceptable": acceptable,
        })

    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> None:
    classes = ["supported", "contradicted", "unaddressed", "weak"]
    total = len(rows)
    if total == 0:
        print("No results.")
        return

    print(f"\n{'=' * 60}")
    print(f"RESULTS  ({total} pairs)")
    print(f"{'=' * 60}")

    # Raw prediction distribution
    pred_counts = Counter(r["predicted"] for r in rows)
    truth_counts = Counter(r["truth"] for r in rows)
    print("\nGround-truth distribution:")
    for c, n in sorted(truth_counts.items()):
        print(f"  {c:<15} {n:>5}  ({n/total:.1%})")

    print("\nPredicted distribution:")
    for c, n in sorted(pred_counts.items()):
        print(f"  {c:<15} {n:>5}  ({n/total:.1%})")

    # Confusion matrix (truth × predicted)
    pred_classes = sorted(set(r["predicted"] for r in rows))
    truth_classes = ["supported", "contradicted", "unaddressed"]
    all_cols = truth_classes + [c for c in pred_classes if c not in truth_classes]

    print(f"\nConfusion matrix  (rows = truth, cols = predicted):")
    header = f"{'':15}" + "".join(f"{c:>14}" for c in all_cols)
    print(header)
    for tc in truth_classes:
        subset = [r for r in rows if r["truth"] == tc]
        preds = Counter(r["predicted"] for r in subset)
        row_str = f"{tc:<15}" + "".join(f"{preds.get(c, 0):>14}" for c in all_cols)
        print(row_str)

    # Per-class precision / recall (strict: only exact match counts)
    print(f"\nPer-class metrics  (strict — exact match only):")
    print(f"  {'Class':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for tc in truth_classes:
        tp = sum(1 for r in rows if r["truth"] == tc and r["predicted"] == tc)
        fp = sum(1 for r in rows if r["truth"] != tc and r["predicted"] == tc)
        fn = sum(1 for r in rows if r["truth"] == tc and r["predicted"] != tc)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {tc:<15} {tp:>6} {fp:>6} {fn:>6} {prec:>8.2%} {rec:>8.2%} {f1:>8.2%}")

    # Acceptable accuracy (WEAK counts for Contradiction and NotMentioned)
    acceptable = sum(1 for r in rows if r["acceptable"])
    print(f"\nAcceptable accuracy (WEAK ok for Contradiction/NotMentioned): "
          f"{acceptable}/{total} = {acceptable/total:.1%}")

    # False-green rate — most dangerous error in a legal tool
    false_green = sum(
        1 for r in rows
        if r["predicted"] == "supported" and r["truth"] != "supported"
    )
    total_non_supported = sum(1 for r in rows if r["truth"] != "supported")
    print(f"False-green rate   (predicted supported, truth ≠ supported):   "
          f"{false_green}/{total_non_supported} = "
          f"{false_green/total_non_supported:.1%}")

    # Missed-supported rate
    missed_supported = sum(
        1 for r in rows
        if r["truth"] == "supported" and r["predicted"] != "supported"
    )
    total_supported = sum(1 for r in rows if r["truth"] == "supported")
    print(f"Missed-support rate (truth supported, predicted ≠ supported):   "
          f"{missed_supported}/{total_supported} = "
          f"{missed_supported/total_supported:.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(split: str, max_docs: int | None) -> None:
    documents, labels = load_split(split)
    if max_docs:
        documents = documents[:max_docs]

    print(f"ContractNLI eval — split={split}, docs={len(documents)}")
    print(f"Total hypothesis pairs: "
          f"{sum(len(d['annotation_sets'][0]['annotations']) for d in documents)}")
    print()

    all_rows: list[dict] = []
    t0 = time.monotonic()

    for i, doc in enumerate(documents):
        n_hyps = len(doc["annotation_sets"][0]["annotations"])
        print(f"[{i+1:>3}/{len(documents)}] doc {doc['id']}  ({n_hyps} hypotheses) ...",
              end="", flush=True)
        t_doc = time.monotonic()
        rows = await eval_document(doc, labels)
        elapsed = time.monotonic() - t_doc

        correct = sum(1 for r in rows if r["acceptable"])
        print(f"  {correct}/{len(rows)} acceptable  ({elapsed:.1f}s)")
        all_rows.extend(rows)

    total_elapsed = time.monotonic() - t0
    print(f"\nCompleted {len(documents)} docs in {total_elapsed:.0f}s")

    compute_metrics(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit to first N documents (default: all)")
    args = parser.parse_args()

    asyncio.run(main(args.split, args.max_docs))
