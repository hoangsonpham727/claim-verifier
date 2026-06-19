"""
Workstream B — learned verdict head over the cached signals.

Trains a multinomial logistic regression on the 5 cached signals plus two
engineered features (gap = p_contra − p_support, mx = max(p_support, p_contra))
and compares it to the hand-tuned tree — all offline, no API calls.

Honest evaluation: out-of-fold predictions via stratified 5-fold CV, so no row
is scored by a model that saw it. To respect the legal-tool constraint we do NOT
argmax: a row is called SUPPORTED only if P(supported) clears a threshold tuned
to hold false-green ≤ cap; otherwise it takes the argmax of the other classes.

  python eval/learn_classifier.py --split dev --max-fg 0.03
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_CLASSES = ["contradicted", "supported", "unaddressed"]  # sorted == sklearn order


def _features(rows) -> np.ndarray:
    ps = np.array([r["p_support"] for r in rows])
    pc = np.array([r["p_contra"] for r in rows])
    ie = np.array([r["inextract"] for r in rows])
    ans = np.array([r["answer_score"] for r in rows])
    rel = np.array([r["relevance"] for r in rows])
    return np.column_stack([ps, pc, ie, ans, rel, pc - ps, np.maximum(ps, pc)])


_FEAT_NAMES = ["p_support", "p_contra", "inextract", "answer_score",
               "relevance", "gap(pc-ps)", "max(ps,pc)"]


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
    return {**{f"f1_{c}": f1[c] for c in _CLASSES},
            "macro_f1": sum(f1.values()) / 3,
            "false_green": fg / n_non if n_non else 0.0,
            "missed_support": missed / n_sup if n_sup else 0.0}


def _apply_threshold(proba, classes, truths, max_fg):
    """SUPPORTED only if P(sup) ≥ τ (smallest τ holding FG ≤ cap); else argmax
    of the non-supported classes."""
    sup_i = list(classes).index("supported")
    p_sup = proba[:, sup_i]
    others = proba.copy()
    others[:, sup_i] = -1.0
    other_pred = classes[np.argmax(others, axis=1)]

    best = None
    for tau in np.unique(np.round(p_sup, 4)):
        preds = np.where(p_sup >= tau, "supported", other_pred)
        fg = np.sum((preds == "supported") & (truths != "supported"))
        n_non = np.sum(truths != "supported")
        if (fg / n_non if n_non else 0.0) <= max_fg:
            best = (tau, preds)
            break
    if best is None:  # cap unreachable; fall back to highest τ
        tau = p_sup.max()
        best = (tau, np.where(p_sup >= tau, "supported", other_pred))
    return best


def main(split: str, max_fg: float) -> None:
    path = Path(__file__).parent / f"scores_{split}.json"
    if not path.exists():
        print(f"No cache at {path}. Run calibrate.py cache first.")
        return
    rows = json.loads(path.read_text())
    X = _features(rows)
    y = np.array([r["truth"] for r in rows], dtype=object)
    print(f"Loaded {len(rows)} rows.  5-fold stratified CV.  FG cap {max_fg:.0%}\n")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = np.zeros((len(rows), 3))
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        classes = clf.classes_
        proba[te] = clf.predict_proba(X[te])
    classes = clf.classes_  # consistent across folds (all classes present)

    tau, preds = _apply_threshold(proba, classes, y, max_fg)
    m = _scorecard(preds, y)

    print(f"Supported-probability threshold (for FG ≤ cap): {tau:.3f}\n")
    print(f"  {'model':<22} {'macro':>6} {'F1_sup':>7} {'F1_con':>7} "
          f"{'F1_un':>6} {'FG':>6} {'miss':>6}")
    print(f"  {'logreg (oof CV)':<22} {m['macro_f1']:>6.3f} "
          f"{m['f1_supported']:>7.3f} {m['f1_contradicted']:>7.3f} "
          f"{m['f1_unaddressed']:>6.3f} {m['false_green']:>6.1%} "
          f"{m['missed_support']:>6.1%}")
    print(f"  {'shipped tree (ref)':<22} {0.467:>6.3f} {0.457:>7.3f} "
          f"{0.317:>7.3f} {0.628:>6.3f} {2.9:>5.1f}% {69.6:>5.1f}%")

    # Interpretability: coefficients from a full-data refit.
    clf.fit(X, y)
    lr = clf.named_steps["logisticregression"]
    print("\nStandardised coefficients (full-data refit):")
    print(f"  {'feature':<14}" + "".join(f"{c[:11]:>13}" for c in lr.classes_))
    for j, name in enumerate(_FEAT_NAMES):
        print(f"  {name:<14}" + "".join(f"{lr.coef_[k, j]:>13.3f}"
                                        for k in range(len(lr.classes_))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--max-fg", type=float, default=0.03)
    main(*vars(ap.parse_args()).values())
