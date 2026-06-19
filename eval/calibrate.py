"""
Phase 6 threshold calibration.

Two stages, decoupled so the expensive API pass runs only once:

  Stage 1 — CACHE (hits the API):
      python eval/calibrate.py cache --split dev
    Runs chunk → rerank → extract → classify_scores per hypothesis over the
    whole split and writes raw signals + ground truth to eval/scores_<split>.json.

  Stage 2 — TUNE (offline, instant, no API):
      python eval/calibrate.py tune --split dev
    Grid-searches (τ_low, τ_con, τ_sup, τ_inex) over the cached scores and reports
    the best threshold sets under a legal-tool objective (minimise false-green,
    then maximise macro-F1).

Re-tune as often as you like; only re-cache when the models or prompts change.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grounding.classify import verdict_from_scores
from grounding.ground import SemchunkRetriever, ground
from grounding.models import Source

_LABEL_MAP = {
    "Entailment":    "supported",
    "Contradiction": "contradicted",
    "NotMentioned":  "unaddressed",
}


def _cache_path(split: str) -> Path:
    return Path(__file__).parent / f"scores_{split}.json"


def load_split(split: str) -> tuple[list[dict], dict]:
    path = Path(__file__).parent / "contract-nli" / f"{split}.json"
    with open(path) as f:
        data = json.load(f)
    return data["documents"], data["labels"]


# ── Stage 1: cache raw scores ─────────────────────────────────────────────────

def _score_one(claim: str, retriever: SemchunkRetriever) -> dict:
    """Run the unified grounding core for one hypothesis; return raw scores.

    `also_reranked=True` additionally classifies on the reranked top-k passages
    so the classifier-input A/B (full-source vs candidates) can be tuned offline.
    """
    g = ground(claim, retriever, top_k=3, classifier_input="source",
               also_reranked=True)
    return {
        "relevance":     round(g.relevance, 4),
        "p_support":     round(g.p_support, 4),     # full source
        "p_contra":      round(g.p_contra, 4),      # full source
        "p_support_rr":  None if g.p_support_rr is None else round(g.p_support_rr, 4),
        "p_contra_rr":   None if g.p_contra_rr is None else round(g.p_contra_rr, 4),
        "inextract":     round(g.inextract, 4),
        "answer_score":  round(g.answer_score, 4),
    }


async def _score_document(doc: dict, labels: dict) -> list[dict]:
    annotations = doc["annotation_sets"][0]["annotations"]
    source_id = str(doc["id"])
    retriever = SemchunkRetriever([Source(id=source_id, text=doc["text"])])

    hyp_ids = list(annotations.keys())

    async def _one(hyp_id: str) -> dict:
        claim = labels[hyp_id]["hypothesis"]
        scores = await asyncio.to_thread(_score_one, claim, retriever)
        scores["doc_id"] = doc["id"]
        scores["hyp_id"] = hyp_id
        scores["truth"] = _LABEL_MAP[annotations[hyp_id]["choice"]]
        return scores

    return await asyncio.gather(*[_one(h) for h in hyp_ids])


async def cache(split: str, max_docs: int | None) -> None:
    documents, labels = load_split(split)
    if max_docs:
        documents = documents[:max_docs]

    n_pairs = sum(len(d["annotation_sets"][0]["annotations"]) for d in documents)
    print(f"Caching scores — split={split}, docs={len(documents)}, pairs={n_pairs}")

    all_rows: list[dict] = []
    t0 = time.monotonic()
    for i, doc in enumerate(documents):
        t_doc = time.monotonic()
        rows = await _score_document(doc, labels)
        all_rows.extend(rows)
        print(f"[{i+1:>3}/{len(documents)}] doc {doc['id']}  "
              f"{len(rows)} pairs  ({time.monotonic()-t_doc:.1f}s)")

    out = _cache_path(split)
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nWrote {len(all_rows)} rows to {out}  ({time.monotonic()-t0:.0f}s)")


# ── Stage 2: offline grid search ──────────────────────────────────────────────

def _metrics(rows: list[dict], taus: dict) -> dict:
    """Apply verdict_from_scores with given thresholds; compute per-class P/R/F1."""
    preds = [
        verdict_from_scores(
            r["p_support"], r["p_contra"], r["inextract"],
            tau_low=taus["low"], tau_con=taus["con"],
            tau_sup=taus["sup"], tau_inex=taus["inex"],
        )[0]
        for r in rows
    ]
    truths = [r["truth"] for r in rows]
    total = len(rows)

    classes = ["supported", "contradicted", "unaddressed"]
    f1 = {}
    for c in classes:
        tp = sum(1 for p, t in zip(preds, truths) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, truths) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, truths) if p != c and t == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    # False-green: predicted supported but truth is not supported
    false_green = sum(
        1 for p, t in zip(preds, truths) if p == "supported" and t != "supported"
    )
    n_non_sup = sum(1 for t in truths if t != "supported")
    fg_rate = false_green / n_non_sup if n_non_sup else 0.0

    macro_f1 = sum(f1.values()) / len(f1)
    return {
        "f1_supported":    f1["supported"],
        "f1_contradicted": f1["contradicted"],
        "f1_unaddressed":  f1["unaddressed"],
        "macro_f1":        macro_f1,
        "false_green":     fg_rate,
        "taus":            taus,
    }


def tune(split: str, max_fg: float) -> None:
    path = _cache_path(split)
    if not path.exists():
        print(f"No cache at {path}. Run:  python eval/calibrate.py cache --split {split}")
        return
    with open(path) as f:
        rows = json.load(f)

    print(f"Loaded {len(rows)} cached rows from {path}")
    print(f"Ground truth: {Counter(r['truth'] for r in rows)}")
    print(f"Objective: maximise macro-F1 subject to false-green ≤ {max_fg:.0%}\n")

    grid_low  = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55]
    grid_con  = [0.6, 0.7, 0.8, 0.85, 0.9]
    grid_sup  = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    grid_inex = [0.5, 0.7, 0.9, 1.0]

    results = []
    for low, con, sup, inex in itertools.product(grid_low, grid_con, grid_sup, grid_inex):
        taus = {"low": low, "con": con, "sup": sup, "inex": inex}
        results.append(_metrics(rows, taus))

    # Feasible set: false-green under cap
    feasible = [r for r in results if r["false_green"] <= max_fg]
    print(f"Grid size: {len(results)}   feasible (fg ≤ {max_fg:.0%}): {len(feasible)}\n")

    def _show(title: str, ranked: list[dict], n: int = 5):
        print(f"── {title} ──")
        print(f"  {'τ_low':>6} {'τ_con':>6} {'τ_sup':>6} {'τ_inex':>6}   "
              f"{'F1_sup':>7} {'F1_con':>7} {'F1_un':>7} {'macro':>7} {'FG':>7}")
        for r in ranked[:n]:
            t = r["taus"]
            print(f"  {t['low']:>6} {t['con']:>6} {t['sup']:>6} {t['inex']:>6}   "
                  f"{r['f1_supported']:>7.3f} {r['f1_contradicted']:>7.3f} "
                  f"{r['f1_unaddressed']:>7.3f} {r['macro_f1']:>7.3f} "
                  f"{r['false_green']:>7.1%}")
        print()

    pool = feasible if feasible else results
    if not feasible:
        print("⚠ No threshold set met the false-green cap; showing best macro-F1 overall.\n")

    _show("Best macro-F1 (within false-green cap)",
          sorted(pool, key=lambda r: -r["macro_f1"]))
    _show("Best contradiction F1 (the hard class)",
          sorted(pool, key=lambda r: -r["f1_contradicted"]))
    _show("Lowest false-green (most conservative)",
          sorted(results, key=lambda r: (r["false_green"], -r["macro_f1"])))

    best = max(pool, key=lambda r: r["macro_f1"])
    t = best["taus"]
    print("Recommended (paste into classify.py):")
    print(f"  _TAU_LOW  = {t['low']}")
    print(f"  _TAU_CON  = {t['con']}")
    print(f"  _TAU_SUP  = {t['sup']}")
    print(f"  _TAU_INEX = {t['inex']}")


# ── Stage 2b: diagnostic — do the raw signals separate the classes? ───────────

def _pctiles(values: list[float]) -> str:
    import numpy as np
    if not values:
        return "(none)"
    a = np.array(values)
    qs = np.percentile(a, [10, 25, 50, 75, 90])
    return (f"mean={a.mean():.3f}  p10={qs[0]:.3f}  p25={qs[1]:.3f}  "
            f"med={qs[2]:.3f}  p75={qs[3]:.3f}  p90={qs[4]:.3f}")


def diagnose(split: str) -> None:
    path = _cache_path(split)
    if not path.exists():
        print(f"No cache at {path}. Run cache first.")
        return
    with open(path) as f:
        rows = json.load(f)

    classes = ["supported", "contradicted", "unaddressed"]
    print(f"Loaded {len(rows)} rows.\n")

    # 1. Per-class signal distributions — does each signal separate the classes?
    for signal in ["p_support", "p_contra", "inextract", "relevance"]:
        print(f"── {signal} by truth class ──")
        for c in classes:
            vals = [r[signal] for r in rows if r["truth"] == c]
            print(f"  {c:<14} {_pctiles(vals)}")
        print()

    # 2. Separation check: for SUPPORTED detection, how much do p_support
    #    distributions for supported vs not-supported overlap?
    sup_pos = sorted(r["p_support"] for r in rows if r["truth"] == "supported")
    sup_neg = sorted(r["p_support"] for r in rows if r["truth"] != "supported")
    print("── SUPPORTED separability (p_support) ──")
    print(f"  supported:     {_pctiles(sup_pos)}")
    print(f"  not-supported: {_pctiles(sup_neg)}")
    # At any τ_sup, false-green = fraction of not-supported with p_support > τ_sup
    print("  false-green floor vs τ_sup (frac of NON-supported above τ):")
    n_neg = len(sup_neg)
    for tau in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        fg = sum(1 for v in sup_neg if v > tau) / n_neg
        rec = sum(1 for v in sup_pos if v > tau) / len(sup_pos)
        print(f"    τ_sup={tau}:  false-green-pop={fg:.1%}   supported-recall={rec:.1%}")
    print()

    # 3. Separation check for CONTRADICTED (the pinned-F1 class)
    con_pos = sorted(r["p_contra"] for r in rows if r["truth"] == "contradicted")
    con_neg = sorted(r["p_contra"] for r in rows if r["truth"] != "contradicted")
    print("── CONTRADICTED separability (p_contra) ──")
    print(f"  contradicted:     {_pctiles(con_pos)}")
    print(f"  not-contradicted: {_pctiles(con_neg)}")
    print("  contra detection vs τ_con:")
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        rec = sum(1 for v in con_pos if v > tau) / len(con_pos)
        fp = sum(1 for v in con_neg if v > tau)
        prec = sum(1 for v in con_pos if v > tau) / max(
            sum(1 for v in con_pos if v > tau) + fp, 1)
        print(f"    τ_con={tau}:  contra-recall={rec:.1%}   contra-precision={prec:.1%}")
    print()

    # 4. Inspect the false-green population at a conservative τ_sup=0.7
    fg_rows = [r for r in rows if r["truth"] != "supported" and r["p_support"] > 0.7]
    print(f"── False-green population (truth≠supported, p_support>0.7): "
          f"{len(fg_rows)} rows ──")
    fg_truth = Counter(r["truth"] for r in fg_rows)
    print(f"  truth breakdown: {dict(fg_truth)}")
    if fg_rows:
        print(f"  their p_contra:  {_pctiles([r['p_contra'] for r in fg_rows])}")
        print(f"  their inextract: {_pctiles([r['inextract'] for r in fg_rows])}")
    print("\n  → If these have LOW p_contra and LOW inextract, no current guard")
    print("    can catch them: the classifier simply over-asserts support.")


# ── Stage 3: report — apply the SHIPPED thresholds from classify.py ───────────

def report(split: str) -> None:
    """Apply the live classify.py constants to the cache; print full scorecard.

    Uses verdict_from_scores with its defaults, which ARE the production
    _TAU_* constants — so this validates the shipped config end-to-end offline.
    """
    from grounding.classify import _TAU_LOW, _TAU_CON, _TAU_SUP, _TAU_INEX

    path = _cache_path(split)
    if not path.exists():
        print(f"No cache at {path}. Run cache first.")
        return
    with open(path) as f:
        rows = json.load(f)

    print(f"Shipped thresholds:  τ_low={_TAU_LOW}  τ_con={_TAU_CON}  "
          f"τ_sup={_TAU_SUP}  τ_inex={_TAU_INEX}")
    print(f"Eval split: {split}  ({len(rows)} pairs)\n")

    preds = [
        verdict_from_scores(r["p_support"], r["p_contra"], r["inextract"])[0]
        for r in rows
    ]
    truths = [r["truth"] for r in rows]

    truth_classes = ["supported", "contradicted", "unaddressed"]
    pred_classes = truth_classes + ["weak"]

    print("Confusion matrix  (rows = truth, cols = predicted):")
    print(f"{'':15}" + "".join(f"{c:>14}" for c in pred_classes))
    for tc in truth_classes:
        idxs = [i for i, t in enumerate(truths) if t == tc]
        counts = Counter(preds[i] for i in idxs)
        print(f"{tc:<15}" + "".join(f"{counts.get(c, 0):>14}" for c in pred_classes))

    print(f"\n  {'Class':<14} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for c in truth_classes:
        tp = sum(1 for p, t in zip(preds, truths) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, truths) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, truths) if p != c and t == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        print(f"  {c:<14} {prec:>8.1%} {rec:>8.1%} {f1:>8.3f}")

    fg = sum(1 for p, t in zip(preds, truths) if p == "supported" and t != "supported")
    n_non_sup = sum(1 for t in truths if t != "supported")
    print(f"\n  False-green rate (pred supported, truth ≠ supported): "
          f"{fg}/{n_non_sup} = {fg/n_non_sup:.1%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cache = sub.add_parser("cache", help="Stage 1 — hit API, write score cache")
    p_cache.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p_cache.add_argument("--max-docs", type=int, default=None)

    p_tune = sub.add_parser("tune", help="Stage 2 — offline grid search over cache")
    p_tune.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p_tune.add_argument("--max-fg", type=float, default=0.03,
                        help="False-green cap (default 0.03 = 3%%)")

    p_diag = sub.add_parser("diagnose", help="Stage 2b — offline signal-separation analysis")
    p_diag.add_argument("--split", default="dev", choices=["train", "dev", "test"])

    p_report = sub.add_parser("report", help="Stage 3 — apply shipped classify.py thresholds to cache")
    p_report.add_argument("--split", default="dev", choices=["train", "dev", "test"])

    args = parser.parse_args()
    if args.cmd == "cache":
        asyncio.run(cache(args.split, args.max_docs))
    elif args.cmd == "tune":
        tune(args.split, args.max_fg)
    elif args.cmd == "diagnose":
        diagnose(args.split)
    elif args.cmd == "report":
        report(args.split)
