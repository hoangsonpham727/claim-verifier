"""
Workstream A — extended decision-tree tuner.

Adds two signals the diagnostics proved discriminative, then grid-searches all
six thresholds under the shipped objective (max macro-F1 s.t. false-green ≤ cap):

  tau_margin  — required (p_contra − p_support) gap for CONTRADICTED
  tau_unaddr  — inextract above this (with low p_support) ⇒ UNADDRESSED

Vectorised with numpy so the 6-D grid stays fast. Mirrors verdict_from_scores
in classify.py exactly (keep in sync).

  python eval/tune2.py --split dev --max-fg 0.03
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grounding.classify import (  # noqa: E402
    _TAU_CON, _TAU_INEX, _TAU_LOW, _TAU_MARGIN, _TAU_SUP, _TAU_UNADDR,
)

_GRID = {
    "low":    [0.2, 0.3, 0.4, 0.45, 0.5, 0.55],
    "con":    [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
    "sup":    [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
    "inex":   [0.5, 0.7, 0.9, 1.0],
    "margin": [0.0, 0.15, 0.2, 0.25, 0.3, 0.35],
    "unaddr": [0.7, 0.8, 0.85, 0.9, 1.0],
}
_SHIPPED = {"low": _TAU_LOW, "con": _TAU_CON, "sup": _TAU_SUP,
            "inex": _TAU_INEX, "margin": _TAU_MARGIN, "unaddr": _TAU_UNADDR}
_CLASSES = ["supported", "contradicted", "unaddressed"]


def _predict(sig, t) -> np.ndarray:
    """Vectorised verdict tree with PER-RULE signal sets. `sig` is a dict with
    arrays for each rule's inputs, so a mode can mix full-source and reranked
    signals rule-by-rule:
      sil  → (ps_sil, pc_sil)  Rule 1 silence (max < τ_low)
      con  → (ps_con, pc_con)  Rule 2 contradiction (margin + τ_con)
      sup  →  ps_sup           Rules 3/3.5 support / inextractable-silence
      ie   →  inextract
    """
    ps_sil, pc_sil = sig["sil"]
    ps_con, pc_con = sig["con"]
    ps_sup = sig["sup"]
    ie = sig["ie"]
    out = np.array(["weak"] * len(ps_sup), dtype=object)

    unaddr1 = np.maximum(ps_sil, pc_sil) < t["low"]
    contra = ~unaddr1 & (pc_con > t["con"]) & ((pc_con - ps_con) > t["margin"])
    sup = ~unaddr1 & ~contra & (ps_sup > t["sup"]) & (ie < t["inex"])
    unaddr2 = ~unaddr1 & ~contra & ~sup & (ie > t["unaddr"]) & (ps_sup < t["sup"])

    out[unaddr1] = "unaddressed"
    out[contra] = "contradicted"
    out[sup] = "supported"
    out[unaddr2] = "unaddressed"
    return out


def _scorecard(preds, truths) -> dict:
    f1 = {}
    for c in _CLASSES:
        tp = int(np.sum((preds == c) & (truths == c)))
        fp = int(np.sum((preds == c) & (truths != c)))
        fn = int(np.sum((preds != c) & (truths == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fg = int(np.sum((preds == "supported") & (truths != "supported")))
    n_non = int(np.sum(truths != "supported"))
    missed = int(np.sum((truths == "supported") & (preds != "supported")))
    n_sup = int(np.sum(truths == "supported"))
    return {
        "f1_supported": f1["supported"],
        "f1_contradicted": f1["contradicted"],
        "f1_unaddressed": f1["unaddressed"],
        "macro_f1": sum(f1.values()) / 3,
        "false_green": fg / n_non if n_non else 0.0,
        "missed_support": missed / n_sup if n_sup else 0.0,
    }


def _row(label, t, m, base=None):
    tau = f"{t['low']}/{t['con']}/{t['sup']}/{t['inex']}/{t['margin']}/{t['unaddr']}"
    d = "" if base is None else f"  (Δmacro {m['macro_f1']-base['macro_f1']:+.3f})"
    return (f"  {label:<26} {tau:>26}  {m['macro_f1']:>6.3f} "
            f"{m['f1_supported']:>7.3f} {m['f1_contradicted']:>7.3f} "
            f"{m['f1_unaddressed']:>6.3f} {m['false_green']:>6.1%} "
            f"{m['missed_support']:>6.1%}{d}")


def _best(sig, truths, max_fg):
    """Grid-search; return (best taus, best scorecard) under the FG cap."""
    results = []
    for combo in itertools.product(*(_GRID[k] for k in _GRID)):
        t = dict(zip(_GRID, combo))
        results.append((t, _scorecard(_predict(sig, t), truths)))
    feasible = [(t, m) for t, m in results if m["false_green"] <= max_fg]
    pool = feasible if feasible else results
    return max(pool, key=lambda r: r[1]["macro_f1"])


def main(split: str, max_fg: float) -> None:
    path = Path(__file__).parent / f"scores_{split}.json"
    if not path.exists():
        print(f"No cache at {path}. Run calibrate.py cache first.")
        return
    rows = json.loads(path.read_text())
    ps = np.array([r["p_support"] for r in rows])
    pc = np.array([r["p_contra"] for r in rows])
    ie = np.array([r["inextract"] for r in rows])
    truths = np.array([r["truth"] for r in rows], dtype=object)
    has_rr = all(r.get("p_support_rr") is not None for r in rows)
    print(f"Loaded {len(rows)} rows.  Objective: max macro-F1 s.t. FG ≤ {max_fg:.0%}")
    print(f"Reranked-candidate signals present: {has_rr}\n")

    hdr = (f"  {'config':<26} {'τ low/con/sup/inex/marg/unadr':>26}  "
           f"{'macro':>6} {'F1_sup':>7} {'F1_con':>7} {'F1_un':>6} {'FG':>6} {'miss':>6}")

    def src_sig():  # full-source for every rule
        return {"sil": (ps, pc), "con": (ps, pc), "sup": ps, "ie": ie}

    base = _scorecard(_predict(src_sig(), _SHIPPED), truths)
    print("── Current shipped tree (full source, shipped τ) ──")
    print(hdr)
    print(_row("shipped", _SHIPPED, base))

    # Classifier-input A/B — each mode grid-tuned independently (Workstream 1).
    modes = [("full-source", src_sig())]
    if has_rr:
        rps = np.array([r["p_support_rr"] for r in rows])
        rpc = np.array([r["p_contra_rr"] for r in rows])
        modes += [
            # all rules read the reranked candidates
            ("candidates", {"sil": (rps, rpc), "con": (rps, rpc), "sup": rps, "ie": ie}),
            # silence from source, all decisions from candidates
            ("hybrid(dec=rr)", {"sil": (ps, pc), "con": (rps, rpc), "sup": rps, "ie": ie}),
            # SURGICAL: candidates only for the contradiction rule; rest full-source
            ("hybrid(con=rr only)", {"sil": (ps, pc), "con": (rps, rpc), "sup": ps, "ie": ie}),
        ]
    else:
        print("\n(⚠ No p_support_rr in cache — re-run `calibrate.py cache` to "
              "compare reranked-candidate input. Showing full-source only.)")

    print("\n── Best per classifier-input mode (within FG cap, grid-tuned) ──")
    print(hdr)
    best_overall = None
    for name, sig in modes:
        t, m = _best(sig, truths, max_fg)
        print(_row(name, t, m, base))
        if best_overall is None or m["macro_f1"] > best_overall[2]["macro_f1"]:
            best_overall = (name, t, m)

    name, t, _ = best_overall
    print(f"\nWinner: {name}.  Recommended (paste into classify.py):")
    for k, kn in [("low", "_TAU_LOW"), ("con", "_TAU_CON"), ("sup", "_TAU_SUP"),
                  ("inex", "_TAU_INEX"), ("margin", "_TAU_MARGIN"),
                  ("unaddr", "_TAU_UNADDR")]:
        print(f"  {kn:<12} = {t[k]}")
    if name != "full-source":
        print(f"  → set GROUNDING_CLASSIFIER_INPUT / _CLASSIFIER_INPUT to '{name.split('(')[0]}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--max-fg", type=float, default=0.03)
    main(*vars(ap.parse_args()).values())
