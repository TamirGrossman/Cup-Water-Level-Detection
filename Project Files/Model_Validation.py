import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, StratifiedGroupKFold
from Model_Builder import build_xgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def cross_validate_model(  
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


# mine - both normal and group kfold
def crosss_val_model(
    X: np.ndarray,
    y: np.ndarray,
    cup_labels,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
    cv: int = 3,
    scoring: str = 'accuracy',
    ):

    cv_splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state)
    model = build_xgb(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                num_class=None,  # let XGBoost infer per fold
            )
    scores = cross_val_score(
        model, X, y,
        cv=cv_splitter, scoring=scoring, n_jobs=3, groups=cup_labels
    )

    print('\nStratified Group K-Fold')
    for fold, score in enumerate(scores):
        print(f"fold: {fold}, cv score: {score}")

    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    model = build_xgb(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                num_class=None,  # let XGBoost infer per fold
            )
    scores = cross_val_score(
        model, X, y,
        cv=cv_splitter, scoring=scoring, n_jobs=3
    )

    print('\nStratified K-Fold')
    for fold, score in enumerate(scores):
        print(f"fold: {fold}, cv score: {score}")
