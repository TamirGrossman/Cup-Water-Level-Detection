"""
Leak-free, cup-balanced cross-validation (with ordinal-aware metrics).

Fixes three flaws that made one fold look much worse than the rest:
  1. Folds stratified on the COMBINED (cup x fill) label -> balanced cup mix.
  2. Scaling + optional feature selection + model live in ONE Pipeline, so each
     fold preprocesses on its own training rows only (no leakage).
  3. Folds are REPEATED so no single unlucky partition dominates.

Metrics: alongside accuracy/f1, two ORDINAL-aware scores are reported because
the fill levels are ordered (20<40<60<80<100) and essentially all errors are
off-by-one neighbours:
  * quadratic_kappa : quadratic-weighted Cohen's kappa (penalises far misses
                      more than near ones; the right metric for ordered classes)
  * adjacent_acc    : fraction predicted within +/-1 fill class
Mean absolute error in CLASS STEPS (lower=better) is printed but not charted.

Return shape matches the original cross_validate_model, so graph_cv_comparison
works unchanged.
"""

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (cohen_kappa_score, get_scorer, make_scorer,
                             mean_absolute_error)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --- ordinal-aware scorers --------------------------------------------------
# NOTE: requires the encoded labels to keep fill order. LabelEncoder sorts
# classes, so 20/40/60/80/100 -> 0/1/2/3/4 and |pred-true| is a class distance.
def _adjacent_acc(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))

_CUSTOM_SCORERS = {
    "quadratic_kappa": make_scorer(cohen_kappa_score, weights="quadratic"),
    "adjacent_acc":    make_scorer(_adjacent_acc),
    "mae_classes":     make_scorer(mean_absolute_error, greater_is_better=False),
}

def _resolve(names: Sequence[str]) -> dict[str, Any]:
    return {n: (_CUSTOM_SCORERS[n] if n in _CUSTOM_SCORERS else get_scorer(n))
            for n in names}


def make_cup_stratified_folds(cup_labels, fill_labels, n_splits=4, n_repeats=5,
                              random_state=42):
    """Folds stratified on the combined (cup, fill) label, repeated n_repeats
    times. Returns a list of (train_idx, test_idx)."""
    cup = np.asarray(cup_labels); fill = np.asarray(fill_labels)
    if len(cup) != len(fill):
        raise ValueError("cup_labels and fill_labels must have the same length")
    strat = np.array([f"{c}|{f}" for c, f in zip(cup, fill)])
    idx = np.arange(len(strat))
    folds = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state + r)
        folds.extend(skf.split(idx, strat))
    return folds


def cross_validate_clean(
    build_model: Callable,
    model_params: dict[str, Any],
    X_raw,
    y,
    cup_labels,
    fill_labels,
    select_k: int | None = None,
    n_splits: int = 4,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: tuple[str, ...] = ("accuracy", "f1_macro",
                                "quadratic_kappa", "adjacent_acc"),
    verbose: bool = True,
) -> dict[str, dict[str, object]]:
    """
    Leak-free, cup-balanced, repeated CV with ordinal-aware metrics.

    Pass RAW (un-scaled, un-selected) features: each fold scales and selects on
    its own training rows. `scoring` are the metrics returned/charted (all
    higher-is-better); MAE-in-class-steps is always also printed.

    build_model is called as build_model(**model_params, random_state=...).
    For a single estimator pass its build fn + params; for the soft-voting
    ensemble pass a builder that returns a VotingClassifier and model_params={}.
    """
    X_raw = np.asarray(X_raw); y = np.asarray(y)

    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if select_k is not None:
        steps.append(("select", SelectKBest(f_classif, k=min(select_k, X_raw.shape[1]))))
    steps.append(("model", build_model(**model_params, random_state=random_state)))
    pipe = Pipeline(steps)

    folds = make_cup_stratified_folds(cup_labels, fill_labels, n_splits,
                                      n_repeats, random_state)

    scorers = _resolve(list(scoring) + ["mae_classes"])
    cv_out = cross_validate(pipe, X_raw, y, cv=folds, scoring=scorers,
                            n_jobs=-1, error_score="raise")

    results = {}
    for metric in scoring:                      # charted metrics only
        s = cv_out[f"test_{metric}"]
        results[metric] = {"scores": s, "mean": float(s.mean()), "std": float(s.std())}
    mae = -cv_out["test_mae_classes"]           # negate (greater_is_better=False)

    if verbose:
        sel = f", ANOVA top-{select_k}" if select_k else ""
        print(f"\n{'=' * 70}")
        print(f"Leak-free CV | (cup x fill)-stratified | "
              f"{n_splits}-fold x {n_repeats} repeats{sel}")
        print(f"{'=' * 70}")
        for metric in scoring:
            s = np.asarray(results[metric]["scores"])
            print(f"{metric:<18}: {s.mean():.4f} +/- {s.std():.4f}  "
                  f"(min={s.min():.4f}, max={s.max():.4f})")
        print(f"{'mae_classes':<18}: {mae.mean():.4f} +/- {mae.std():.4f}  "
              f"(lower is better; avg fill-buckets off)")
        print(f"{'=' * 70}")

    return results