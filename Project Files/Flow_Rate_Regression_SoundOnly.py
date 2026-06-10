"""
Flow-rate REGRESSION pipeline.

Unlike the classification pipeline (which predicts a discrete fill class and
then back-computes flow rate with a fixed formula), this trains models to
predict the continuous water flow rate (mL/s) DIRECTLY from the audio
features. The flow rate is a real learned target here, not arithmetic applied
to a class label.

Feature setup: "sound only".
  - All per-window acoustic features are used.
  - `duration_s` is DROPPED from the features (it is only used to build the
    target). The model must infer flow rate from the pouring sound alone.
  - The explicit `cup` id column is also dropped from the features (the cup's
    acoustic signature is already present in the sound); it is only used to
    build the target. The `fill` column is dropped (it defines the target).

This is the harder, more honest test of "does the sound carry flow-rate
information" than the sound + duration variant, and it will usually score
lower because the pour time is no longer handed to the model.

Target:
    flow_rate_mL_s = (fill / 100) * cup_capacity_mL / duration_s

Models: XGBRegressor + RandomForestRegressor, averaged 50/50 (same ensemble
idea as the classifier, but averaging predicted numbers instead of class
probabilities).

Run directly:  python Flow_Rate_Regression_SoundOnly.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_and_build_target(
    csv_path: str,
    cup_sizes_ml: dict[int, float] = CUP_SIZES_ML,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load the CSV, build the continuous flow-rate target, and assemble the
    "sound + duration" feature matrix.

    Returns:
        X            : feature DataFrame (sound features + duration_s)
        y            : flow rate (mL/s), continuous target
        fill_labels  : the true fill % per row (kept aside, for per-class plots)
        cup_labels   : the cup id per row (kept aside, for reference)
        feature_names
    """
    data = pd.read_csv(csv_path)

    # Case-insensitive column resolution (handles 'cup'/'Cup', 'fill'/'Fill').
    lookup = {c.lower(): c for c in data.columns}
    for col in ("cup", "fill", "duration_s"):
        if col not in lookup:
            raise ValueError(f"CSV is missing required column '{col}'")
    cup_col, fill_col, dur_col = lookup["cup"], lookup["fill"], lookup["duration_s"]

    cup = data[cup_col].astype(int)
    fill = data[fill_col].astype(float)
    duration_s = data[dur_col].astype(float)

    if (duration_s <= 0).any():
        raise ValueError("found non-positive duration_s; cannot form flow rate")

    cup_size = cup.map(cup_sizes_ml)
    if cup_size.isna().any():
        bad = sorted(cup[cup_size.isna()].unique())
        raise ValueError(f"cup id(s) {bad} missing from cup_sizes_ml")

    # Continuous target: flow rate in mL/s.
    y = ((fill / 100.0) * cup_size / duration_s).to_numpy(dtype=float)

    # Features = sound only. Drop the target-defining columns AND duration_s
    # (duration is used to build the target but is NOT given to the model).
    # Also drop the explicit cup id (sound already encodes the cup).
    X = data.drop(columns=[cup_col, fill_col, dur_col])
    # Drop constant columns (no information).
    X = X.drop(columns=X.columns[X.nunique() < 2])

    return X, y, fill.to_numpy(), cup.to_numpy(), X.columns.tolist()


def split_with_indices(X, y, test_size: float = 0.2, random_state: int = 42):
    """Random train/test split that also returns positional row indices."""
    idx = np.arange(len(X))
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
        X, y, idx, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"Training set: {X_tr.shape[0]} samples")
    print(f"Test set:     {X_te.shape[0]} samples")
    return X_tr, X_te, y_tr, y_te, np.asarray(tr_idx), np.asarray(te_idx)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_xgb_regressor(random_state: int = 42, **kwargs) -> xgb.XGBRegressor:
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        verbosity=0,
        random_state=random_state,
    )
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def build_rf_regressor(random_state: int = 42, **kwargs) -> RandomForestRegressor:
    params = dict(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )
    params.update(kwargs)
    return RandomForestRegressor(**params)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(name: str, m: dict[str, float]) -> None:
    print(f"{name:<18} MAE={m['MAE']:.4f} mL/s   "
          f"RMSE={m['RMSE']:.4f} mL/s   R2={m['R2']:.4f}")


def cross_validate_regressor(model, X, y, cv: int = 5, random_state: int = 42):
    """Plain K-Fold CV (no stratification for a continuous target)."""
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    r2 = cross_val_score(model, X, y, cv=splitter, scoring="r2", n_jobs=-1)
    rmse = -cross_val_score(model, X, y, cv=splitter,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"  CV R2:   {r2.mean():.4f} +/- {r2.std():.4f}")
    print(f"  CV RMSE: {rmse.mean():.4f} +/- {rmse.std():.4f} mL/s")
    return {"r2": r2, "rmse": rmse}


# ---------------------------------------------------------------------------
# Per-fill-class breakdown (for reporting/plots)
# ---------------------------------------------------------------------------
def per_class_table(
    test_fill: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Mean true vs predicted flow rate grouped by the true fill class."""
    df = pd.DataFrame({
        "fill_pct": test_fill,
        "true_flow_rate_mL_s": y_true,
        "pred_flow_rate_mL_s": y_pred,
    })
    return (df.groupby("fill_pct")[["true_flow_rate_mL_s", "pred_flow_rate_mL_s"]]
              .mean().reset_index().sort_values("fill_pct").reset_index(drop=True))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def graph_flow_rate_regression(per_class: pd.DataFrame,
                               y_true: np.ndarray, y_pred: np.ndarray,
                               test_fill: np.ndarray, figsize=(13, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # (1) Mean flow rate by fill class: true vs predicted.
    classes = per_class["fill_pct"].tolist()
    x = range(len(classes))
    w = 0.38
    ax1.bar([i - w / 2 for i in x], per_class["true_flow_rate_mL_s"],
            width=w, label="True", color="#4C72B0")
    ax1.bar([i + w / 2 for i in x], per_class["pred_flow_rate_mL_s"],
            width=w, label="Predicted", color="#DD8452")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{c:.0f}%" for c in classes])
    ax1.set_xlabel("Fill class")
    ax1.set_ylabel("Mean flow rate (mL/s)")
    ax1.set_title("Mean Flow Rate by Fill Class: True vs Predicted")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # (2) Predicted vs true per sample, with y = x reference line.
    sc = ax2.scatter(y_true, y_pred, c=test_fill, cmap="viridis",
                     alpha=0.85, edgecolor="k", linewidth=0.3)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1,
             label="y = x (perfect)")
    ax2.set_xlabel("True flow rate (mL/s)")
    ax2.set_ylabel("Predicted flow rate (mL/s)")
    ax2.set_title("Predicted vs True Flow Rate (per sample)")
    ax2.legend()
    ax2.grid(linestyle="--", alpha=0.4)
    cbar = fig.colorbar(sc, ax=ax2, ticks=sorted(np.unique(test_fill)))
    cbar.set_label("Fill %")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def run(csv_path: str, random_state: int = 42, run_cv: bool = True, make_plots: bool = True):
    # 1. Data + continuous target
    X, y, fill_labels, cup_labels, features = load_and_build_target(csv_path)
    print(f"Features: {X.shape[1]} (sound only, no duration) | Samples: {X.shape[0]}")
    print(f"Flow rate target -> min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f} mL/s\n")

    # 2. Split (keep indices so we can recover the fill class of test rows)
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = split_with_indices(X, y, test_size=0.2,
                                                                random_state=random_state)
    test_fill = fill_labels[te_idx]

    # 3. Scale
    X_tr_s, X_te_s, scaler = scale_features(X_tr, X_te)

    # 4. Build models
    xgb_model = build_xgb_regressor(random_state=random_state)
    rf_model = build_rf_regressor(random_state=random_state)

    # 5. Cross-validation (on training data only)
    if run_cv:
        print("\nXGB regressor — K-Fold CV:")
        cross_validate_regressor(xgb_model, X_tr_s, y_tr, cv=5, random_state=random_state)
        print("RandomForest regressor — K-Fold CV:")
        cross_validate_regressor(rf_model, X_tr_s, y_tr, cv=5, random_state=random_state)

    # 6. Train
    xgb_model.fit(X_tr_s, y_tr)
    rf_model.fit(X_tr_s, y_tr)

    # 7. Predict on the test set
    pred_xgb = xgb_model.predict(X_te_s)
    pred_rf = rf_model.predict(X_te_s)
    pred_ens = 0.5 * pred_xgb + 0.5 * pred_rf   # 50/50 ensemble of predicted flow rates

    # 8. Report
    print(f"\n{'='*64}\nTest-set flow-rate prediction (mL/s)\n{'='*64}")
    print_metrics("XGBoost", regression_metrics(y_te, pred_xgb))
    print_metrics("RandomForest", regression_metrics(y_te, pred_rf))
    print_metrics("Ensemble (50/50)", regression_metrics(y_te, pred_ens))
    print(f"{'='*64}")

    per_class = per_class_table(test_fill, y_te, pred_ens)
    print("\nMean flow rate by fill class (ensemble), across all cups:")
    print(per_class.to_string(index=False))

    # 9. Plots
    if make_plots:
        graph_flow_rate_regression(per_class, y_te, pred_ens, test_fill)

    return {
        "xgb": xgb_model, "rf": rf_model, "scaler": scaler,
        "y_true": y_te, "y_pred_ensemble": pred_ens,
        "per_class": per_class, "test_indices": te_idx,
    }


if __name__ == "__main__":
    CSV_PATH = r"C:\Users\amitn\OneDrive\Desktop\Ml\Cup-Water-Level-Detection-main\Project Files\cup_dataset.csv"
    run(CSV_PATH)
