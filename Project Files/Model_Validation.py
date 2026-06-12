import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, StratifiedGroupKFold
from Model_Builder import build_xgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def cross_validate_model_old(  
    model: xgb.XGBClassifier | RandomForestClassifier, 
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    cv: int = 5,
    scoring: tuple[str, ...] = ('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'),
    verbose: bool = True,
) -> dict[str, object]:
    """
    Evaluate the XGBoost classifier with Stratified k-Fold cross-validation.

    K-Fold cross-validation splits the data into `cv` equally sized folds.
    Each fold is used once as a validation set while the remaining k-1 folds
    form the training set, so every sample is validated exactly once. The
    *stratified* variant preserves the class proportions in every fold, which
    matters for (potentially imbalanced) multi-class problems like this one.

    This gives a more robust estimate of generalization performance than a
    single train/test split, along with a measure of variance across folds.

    Parameters:
    -----------
    X : np.ndarray
        Scaled feature matrix (n_samples, n_features). If feature selection is
        used, pass the already-reduced matrix so CV reflects the final model.
    y : np.ndarray
        Encoded labels.
    n_estimators, learning_rate, random_state :
        Hyperparameters for the XGBoost estimator (matched to final training).
    cv : int
        Number of folds k (default: 5).
    scoring : tuple of str
        sklearn scoring metrics to compute across the folds.
    verbose : bool
        Print a per-metric summary (mean +/- std and per-fold scores).

    Returns:
    --------
    dict
        {metric: {'scores': np.ndarray, 'mean': float, 'std': float}} for each
        requested metric.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Stratified {cv}-Fold Cross-Validation")
        print(f"{'='*60}")

    results = {}
    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=metric, n_jobs=-1)
        results[metric] = {
            'scores': scores,
            'mean': float(scores.mean()),
            'std': float(scores.std()),
        }
        if verbose:
            fold_str = ", ".join(f"{s:.4f}" for s in scores)
            print(f"{metric:<18}: {scores.mean():.4f} +/- {scores.std():.4f} "
                  f"(folds: [{fold_str}])")

    if verbose:
        print(f"{'='*60}")

    return results

from sklearn.base import clone
from sklearn.metrics import get_scorer
from Data_Loader import scale_features

def cross_validate_model(
    model: xgb.XGBClassifier | RandomForestClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42,
    cv: int = 5,
    scoring: tuple[str, ...] = ('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'),
    verbose: bool = True,
) -> dict[str, object]:
    """
    Evaluate the classifier with Stratified k-Fold cross-validation, scaling
    each fold independently via `scale_features` (scaler is fit on the
    training fold only, then applied to both train and validation folds, to
    avoid data leakage).

    Parameters:
    -----------
    X : pd.DataFrame
        Unscaled feature matrix (n_samples, n_features). Scaling is performed
        per fold inside this function.
    y : pd.Series
        Encoded labels.
    cv : int
        Number of folds k (default: 5).
    scoring : tuple of str
        sklearn scoring metric names to compute across the folds.
    verbose : bool
        Print a per-metric summary (mean +/- std and per-fold scores).

    Returns:
    --------
    dict
        {metric: {'scores': np.ndarray, 'mean': float, 'std': float}} for each
        requested metric.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Stratified {cv}-Fold Cross-Validation (per-fold scaling)")
        print(f"{'='*60}")

    scorers = {metric: get_scorer(metric) for metric in scoring}
    fold_scores: dict[str, list[float]] = {metric: [] for metric in scoring}

    # StratifiedKFold.split returns positional indices, so use .iloc
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit scaler on the training fold only, apply to both splits
        X_train_scaled, X_val_scaled, _ = scale_features(X_train, X_val)

        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train)

        for metric in scoring:
            score = scorers[metric](fold_model, X_val_scaled, y_val)
            fold_scores[metric].append(score)

        if verbose:
            fold_summary = ", ".join(
                f"{metric}={fold_scores[metric][-1]:.4f}" for metric in scoring
            )
            print(f"Fold {fold_idx}/{cv}: {fold_summary}")

    results = {}
    for metric in scoring:
        scores = np.array(fold_scores[metric])
        results[metric] = {
            'scores': scores,
            'mean': float(scores.mean()),
            'std': float(scores.std()),
        }
        if verbose:
            fold_str = ", ".join(f"{s:.4f}" for s in scores)
            print(f"{metric:<18}: {scores.mean():.4f} +/- {scores.std():.4f} "
                  f"(folds: [{fold_str}])")

    if verbose:
        print(f"{'='*60}")

    return results

def kfold_cv_multiple(model, X, y, k=5, scoring_metrics=None):
    """
    Perform k-fold CV with multiple scoring metrics.
    
    Parameters:
    -----------
    model : estimator object
    X : array-like
    y : array-like
    k : int, default=5
    scoring_metrics : list, default=['accuracy', 'precision', 'recall', 'f1']
    
    Returns:
    --------
    dict : Results for each metric
    """
    
    if scoring_metrics is None:
        scoring_metrics = scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',  # Fixed: use weighted average
            'recall': 'recall_weighted',        # Fixed: use weighted average
            'f1': 'f1_weighted'                 # Fixed: use weighted average
        }
    
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring_metrics)
    
    results = {}
    for metric in scoring_metrics:
        scores = cv_results[f'test_{metric}']
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    return results, cv_results

