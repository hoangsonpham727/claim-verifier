"""
Pipeline redundancy ablation — does each component move the ContractNLI numbers?

The verdict (verdict_from_scores in classify.py) consumes only three signals:
  p_support, p_contra  → kanon-universal-classifier  (two calls)
  inextract            → kanon-answer-extractor       (reranker picks its input)
The reranker's `relevance` and the extractor's `answer_score` are cached but
never feed the verdict.  So we can ablate the extractor and the second
(negation) classifier call entirely OFFLINE, by neutralising the corresponding
signal across the cached dev scores and re-tuning thresholds for a fair fight.

  python eval/ablate.py                 # full comparison table (dev)
  python eval/ablate.py --split dev --max-fg 0.03

Each ablation gets its OWN best thresholds (grid search under the shipped
objective: maximise macro-F1 s.t. false-green ≤ cap), so we never penalise an
ablation for thresholds tuned to the full signal set.

The embedder ablation (B1) is end-to-end and lives in run_eval.py via
GROUNDING_NO_EMBED — see the plan / README, not here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grounding.classify import (  # noqa: E402
    _TAU_CON,
    _TAU_INEX,
    _TAU_LOW,
    _TAU_SUP,
    verdict_from_scores,
)

# Same grid as eval/calibrate.py tune() — keep in sync.
_GRID_LOW = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55]
_GRID_CON = [0.6, 0.7, 0.8, 0.85, 0.9]
_GRID_SUP = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
_GRID_INEX = [0.5, 0.7, 0.9, 1.0]

_SHIPPED = {"low": _TAU_LOW, "con": _TAU_CON, "sup": _TAU_SUP, "inex": _TAU_INEX}


def _cache_path(split: str) -> Path:
    return Path(__file__).parent / f"scores_{split}.json"


def score(rows: list[dict], taus: dict) -> dict:
    """Apply verdict_from_scores at `taus`; return the full scorecard."""
    preds = [
        verdict_from_scores(
            r["p_support"], r["p_contra"], r["inextract"],
            tau_low=taus["low"], tau_con=taus["con"],
            tau_sup=taus["sup"], tau_inex=taus["inex"],
        )[0]
        for r in rows
    ]
    truths = [r["truth"] for r in rows]

    classes = ["supported", "contradicted", "unaddressed"]
    f1 = {}
    for c in classes:
        tp = sum(1 for p, t in zip(preds, truths) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, truths) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, truths) if p != c and t == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    fg = sum(1 for p, t in zip(preds, truths) if p == "supported" and t != "supported")
    n_non_sup = sum(1 for t in truths if t != "supported")
    fg_rate = fg / n_non_sup if n_non_sup else 0.0

    missed = sum(1 for p, t in zip(preds, truths) if t == "supported" and p != "supported")
    n_sup = sum(1 for t in truths if t == "supported")
    missed_rate = missed / n_sup if n_sup else 0.0

    return {
        "f1_supported":    f1["supported"],
        "f1_contradicted": f1["contradicted"],
        "f1_unaddressed":  f1["unaddressed"],
        "macro_f1":        sum(f1.values()) / len(f1),
        "false_green":     fg_rate,
        "missed_support":  missed_rate,
        "taus":            taus,
    }


def tune_grid(rows: list[dict], max_fg: float) -> dict:
    """Grid-search thresholds; return the best scorecard under the false-green cap."""
    results = [
        score(rows, {"low": low, "con": con, "sup": sup, "inex": inex})
        for low, con, sup, inex in itertools.product(
            _GRID_LOW, _GRID_CON, _GRID_SUP, _GRID_INEX
        )
    ]
    feasible = [r for r in results if r["false_green"] <= max_fg]
    pool = feasible if feasible else results
    best = max(pool, key=lambda r: r["macro_f1"])
    best["_feasible"] = bool(feasible)
    return best


# ── Ablation transforms (neutralise the signal a component contributes) ───────

def _ablate(rows: list[dict], **overrides) -> list[dict]:
    return [{**r, **overrides} for r in rows]


# (label, transformed rows, one-line justification)
def _configs(rows: list[dict]) -> list[tuple[str, list[dict], str]]:
    return [
        ("baseline", rows,
         "all signals (universal-classifier x2 + answer-extractor)"),
        ("A1: no extractor (inextract=0)", _ablate(rows, inextract=0.0),
         "drop kanon-answer-extractor → Rule 3 inextract guard removed"),
        ("A2: no contradiction call (p_contra=0)", _ablate(rows, p_contra=0.0),
         "drop 2nd universal-classifier call → no CONTRADICTED rule"),
    ]


def _print_table(title: str, rows_by_cfg: list[tuple[str, dict]]) -> None:
    print(f"\n── {title} ──")
    hdr = (f"  {'config':<40} {'τ(low/con/sup/inex)':>22}  "
           f"{'macro':>6} {'F1_sup':>7} {'F1_con':>7} {'F1_un':>6} "
           f"{'FG':>6} {'miss':>6}")
    print(hdr)
    base = rows_by_cfg[0][1]
    for label, m in rows_by_cfg:
        t = m["taus"]
        tau_s = f"{t['low']}/{t['con']}/{t['sup']}/{t['inex']}"
        d = m["macro_f1"] - base["macro_f1"]
        delta = "" if label == rows_by_cfg[0][0] else f"  (Δmacro {d:+.3f})"
        flag = "" if m.get("_feasible", True) else " ⚠fg-cap-missed"
        print(f"  {label:<40} {tau_s:>22}  "
              f"{m['macro_f1']:>6.3f} {m['f1_supported']:>7.3f} "
              f"{m['f1_contradicted']:>7.3f} {m['f1_unaddressed']:>6.3f} "
              f"{m['false_green']:>6.1%} {m['missed_support']:>6.1%}{delta}{flag}")


def main(split: str, max_fg: float) -> None:
    path = _cache_path(split)
    if not path.exists():
        print(f"No cache at {path}. Run:  python eval/calibrate.py cache --split {split}")
        return
    with open(path) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} cached rows from {path}")
    print(f"Objective: maximise macro-F1 subject to false-green ≤ {max_fg:.0%}")

    configs = _configs(rows)

    # View 1 — re-tuned: each ablation gets its own best thresholds (fair fight).
    retuned = [(label, tune_grid(crows, max_fg)) for label, crows, _ in configs]
    _print_table("Re-tuned per ablation (each config's own best thresholds)", retuned)

    # View 2 — shipped thresholds held fixed: what happens to the LIVE config.
    fixed = [(label, score(crows, _SHIPPED)) for label, crows, _ in configs]
    _print_table("At shipped thresholds (held fixed)", fixed)

    # Verdict per ablation, against the decision criteria.
    print("\n── Interpretation (Δmacro within ±0.01 and FG ≤ cap ⇒ redundant for accuracy) ──")
    base_macro = retuned[0][1]["macro_f1"]
    for (label, crows, why), (_, m) in list(zip(configs, retuned))[1:]:
        d = m["macro_f1"] - base_macro
        redundant = abs(d) <= 0.01 and m["false_green"] <= max_fg
        verdict = "REDUNDANT for benchmark accuracy" if redundant else "MATERIAL — keep"
        print(f"  • {label}")
        print(f"      {why}")
        print(f"      Δmacro={d:+.3f}  FG={m['false_green']:.1%}  "
              f"F1_con={m['f1_contradicted']:.3f}  →  {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--max-fg", type=float, default=0.03,
                    help="False-green cap (default 0.03 = 3%%)")
    args = ap.parse_args()
    main(args.split, args.max_fg)
