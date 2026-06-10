"""
Flow-rate estimation by REGRESSION.

This replaces the old formula-based approach (classifier predicts a fill class,
flow rate back-computed by arithmetic). Here the flow rate is a LEARNED target:
regression models are trained to predict the continuous flow rate (mL/s)
directly from the audio features + pour duration.

`compute_flow_rates` does the whole thing internally:
  1. Build the continuous target  flow = (fill/100) * cup_capacity / duration.
  2. Feature set = "sound + duration" (drop the explicit cup id and fill).
  3. Split using the SAME test rows as the main pipeline (`test_indices`), so
     the reported flow rate is on the same held-out set as the classifier.
  4. Tune + cross-validate an XGBoost regressor AND a RandomForest regressor,
     then pick whichever model+hyperparameters score best (CV R^2).
  5. Predict flow rate on the test rows with that best regressor.

Returns a dict compatible with Graph_results.graph_flow_rate (keys 'per_class'
and 'per_sample'), plus the winning model name, its tuned params, and metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


# Physical cup capacities in milliliters, keyed by the 'cup' id in the CSV.
CUP_SIZES_ML: dict[int, float] = {
    1: 300.0,
    2: 350.0,
    3: 400.0,
}

# Hyperparameter grids searched to find the best-suited regressor.
XGB_PARAM_GRID = {
    "n_estimators":  [200, 300],
    "max_depth":     [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.6, 0.8],
}
FOREST_PARAM_GRID = {
    "n_estimators": [200, 300],
    "max_depth":    [None, 10],
    "max_features": ["sqrt", 0.5],
}


def _load_target_and_features(
    csv_path: str, cup_sizes_ml: dict[int, float]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV, build the continuous flow-rate target and the feature matrix."""
    data = pd.read_csv(csv_path)

    # Case-insensitive column resolution ('cup'/'Cup', 'fill'/'Fill').
    lookup = {c.lower(): c for c in data.columns}
    for col in ("cup", "fill", "duration_s"):
        if col not in lookup:
            raise ValueError(f"CSV is missing required column '{col}'")
    cup_col, fill_col, dur_col = lookup["cup"], lookup["fill"], lookup["duration_s"]

    cup = data[cup_col].astype(int)
    fill = data[fill_col].astype(float)
    duration = data[dur_col].astype(float)

    if (duration <= 0).any():
        raise ValueError("found non-positive duration_s; cannot form flow rate")
    cup_size = cup.map(cup_sizes_ml)
    if cup_size.isna().any():
        bad = sorted(cup[cup_size.isna()].unique())
        raise ValueError(f"cup id(s) {bad} missing from cup_sizes_ml")

    # Continuous target: flow rate in mL/s.
    y = ((fill / 100.0) * cup_size / duration).to_numpy(dtype=float)

    # Features = sound + duration. Drop the explicit cup id (the sound already
    # encodes the cup) and fill (it defines the target). Drop constant columns.
    X = data.drop(columns=[cup_col, fill_col])
    X = X.drop(columns=X.columns[X.nunique() < 2])

    return X, y, fill.to_numpy(), cup.to_numpy()


def _tune_regressor(kind: str, X, y, cv_splitter, random_state: int):
    """Grid-search one regressor type; return (best_estimator, params, cv_r2)."""
    if kind == "xgb":
        base = xgb.XGBRegressor(
            objective="reg:squarederror", n_jobs=-1, verbosity=0,
            random_state=random_state,
        )
        grid = XGB_PARAM_GRID
    else:
        base = RandomForestRegressor(n_jobs=-1, random_state=random_state)
        grid = FOREST_PARAM_GRID

    search = GridSearchCV(base, grid, scoring="r2", cv=cv_splitter, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def compute_flow_rates(
    csv_path: str,
    test_indices: np.ndarray | list[int],
    tune: bool = True,
    cv: int = 5,
    random_state: int = 42,
    cup_sizes_ml: dict[int, float] = CUP_SIZES_ML,
    verbose: bool = True,
    **_ignored,                 # absorbs old args (e.g. y_pred/encoder) harmlessly
) -> dict[str, object]:
    """
    Train + tune regression models to predict flow rate, pick the best-suited
    one, and report predicted vs true flow rate on the pipeline's test rows.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV (needs 'cup', 'fill', 'duration_s').
    test_indices : array-like of int
        Row positions that make up the test set (from the main pipeline split).
        Training uses every other row.
    tune : bool
        If True, grid-search hyperparameters for each regressor. If False, use
        sensible defaults and only cross-validate them.
    cv : int
        K-Fold splits for tuning / CV scoring.

    Returns
    -------
    dict with: 'per_class', 'per_sample' (for plotting), 'mean_flow_rate_true',
    'mean_flow_rate_predicted', 'best_model', 'best_params', 'metrics',
    'cv_scores'.
    """
    test_indices = np.asarray(test_indices)
    X, y, fill_all, cup_all = _load_target_and_features(csv_path, cup_sizes_ml)

    # Same held-out test rows as the classifier; train on the rest.
    test_pos = np.sort(test_indices)
    train_pos = np.setdiff1d(np.arange(len(X)), test_pos)

    X_tr, X_te = X.iloc[train_pos], X.iloc[test_pos]
    y_tr, y_te = y[train_pos], y[test_pos]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # --- Find the best-suited model + hyperparameters among XGB / RF ---
    candidates: dict[str, tuple] = {}
    for kind in ("xgb", "forest"):
        if tune:
            model, params, cv_r2 = _tune_regressor(kind, X_tr_s, y_tr, cv_splitter, random_state)
        else:
            model = (xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300,
                                      max_depth=5, learning_rate=0.05, subsample=0.6,
                                      n_jobs=-1, verbosity=0, random_state=random_state)
                     if kind == "xgb"
                     else RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                                n_jobs=-1, random_state=random_state))
            cv_r2 = float(cross_val_score(model, X_tr_s, y_tr, cv=cv_splitter,
                                          scoring="r2", n_jobs=-1).mean())
            params = {}
        candidates[kind] = (model, params, cv_r2)

    best_kind = max(candidates, key=lambda k: candidates[k][2])
    best_model, best_params, best_cv_r2 = candidates[best_kind]
    best_label = "XGBoost" if best_kind == "xgb" else "RandomForest"

    # Fit the winner on the full training set and predict the test rows.
    best_model.fit(X_tr_s, y_tr)
    pred = best_model.predict(X_te_s)

    metrics = {
        "MAE": float(mean_absolute_error(y_te, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "R2": float(r2_score(y_te, pred)),
    }

    # Per-sample table (test rows) — keep 'cup' for the scatter coloring.
    per_sample = pd.DataFrame({
        "csv_row": test_pos,
        "cup": cup_all[test_pos],
        "fill_pct": fill_all[test_pos],
        "true_flow_rate_mL_s": y_te,
        "pred_flow_rate_mL_s": pred,
    })

    # Mean flow rate per fill class (20/40/...), pooled across all cups.
    per_class = (
        per_sample
        .groupby("fill_pct")[["true_flow_rate_mL_s", "pred_flow_rate_mL_s"]]
        .mean().reset_index().sort_values("fill_pct").reset_index(drop=True)
    )

    mean_true = float(np.mean(y_te))
    mean_pred = float(np.mean(pred))

    if verbose:
        print(f"\n{'=' * 64}")
        print("Flow-rate regression — best-suited model selection")
        print(f"{'=' * 64}")
        for kind, (_, params, cv_r2) in candidates.items():
            tag = "  <-- chosen" if kind == best_kind else ""
            name = "XGBoost" if kind == "xgb" else "RandomForest"
            print(f"{name:<14} CV R2={cv_r2:.4f}  params={params}{tag}")
        print(f"{'-' * 64}")
        print(f"Best model: {best_label}")
        print(f"Test-set:   MAE={metrics['MAE']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}  (mL/s)")
        print(f"{'-' * 64}")
        print(f"{'fill %':<10}{'TRUE':>14}{'PREDICTED':>14}")
        for _, r in per_class.iterrows():
            print(f"{r['fill_pct']:<10.0f}"
                  f"{r['true_flow_rate_mL_s']:>14.4f}"
                  f"{r['pred_flow_rate_mL_s']:>14.4f}")
        print(f"{'=' * 64}")

    return {
        "mean_flow_rate_true": mean_true,
        "mean_flow_rate_predicted": mean_pred,
        "per_class": per_class,
        "per_sample": per_sample,
        "best_model": best_label,
        "best_params": best_params,
        "best_cv_r2": best_cv_r2,
        "metrics": metrics,
        "cv_scores": {k: v[2] for k, v in candidates.items()},
        "model": best_model,
        "scaler": scaler,
    }
