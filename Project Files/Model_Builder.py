from typing import Any, Callable

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

def build_xgb(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth=6,
    random_state: int = 42,
    num_class: int | None = 5,
    verbosity: int = 0,
    subsample: float =0.4,
    colsample_bytree: float = 0.55,
    objective: str = 'multi:softprob',
    n_jobs: int =5,
    reg_alpha: float = 0.5,
    reg_lambda: float = 0.3
) -> xgb.XGBClassifier:
    """
    Construct an (untrained) XGBoost classifier with the project's
    default hyperparameters.
    """

    '''
    # Advanced tuning example
    model = xgb.XGBClassifier(
        n_estimators=200,           # More boosting rounds
        max_depth=5,                # Shallower trees reduce overfitting
        learning_rate=0.05,         # Lower learning rate for stability
        subsample=0.7,              # Use 70% of samples per tree
        colsample_bytree=0.7,       # Use 70% of features per tree
        min_child_weight=1,         # Minimum sample weight in child node
        gamma=0,                    # L1 regularization
        reg_lambda=1,               # L2 regularization
        reg_alpha=0,                # L1 weight regularization
        objective='multi:softprob', # 5-class multiclass
        num_class=5,                # Number of classes
        random_state=42
    )
    '''

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth= max_depth,
        learning_rate= learning_rate,
        subsample= subsample,
        colsample_bytree= colsample_bytree,
        objective= objective,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=verbosity,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        num_class = num_class
        )

def build_randomforest(
    n_estimators: int = 100,
    max_depth: int = 6,
    random_state: int = 42,
    n_jobs: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
    max_features: str | None = 'log2',
    verbosity: int = 0,
    ) -> RandomForestClassifier:
    "Builds an untraineds random forest model"
    model_Forest = RandomForestClassifier(
        n_estimators= n_estimators,
        max_depth= max_depth,
        max_features= max_features,
        random_state=random_state, 
        verbose= verbosity,
        n_jobs=n_jobs,
        min_samples_split= min_samples_split,
        min_samples_leaf= min_samples_leaf,
        class_weight= class_weight
        )
    return model_Forest


def forward_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    model_params: dict[str, Any], 
    build_model: Callable,  
    random_state: int = 42,
    max_features: int | None = None,
    cv: int = 5,
    scoring: str = 'accuracy',
    initial_features: list[str] = [],
    min_improvement: float = 1e-6,
    verbose: bool = True,
) -> tuple[list[int], list[str], list[dict[str, object]]]:
    """
    Forward Feature Selection (FFS) -- a *wrapper* method.

    Greedy procedure described in the course notes:
      1. Start from an EMPTY feature set (k = 0).
      2. At each step, try adding each not-yet-selected feature and score the
         resulting subset using the actual classifier, evaluated with
         cross-validation.
      3. Permanently add the single feature that improves the score the most.
      4. Stop when no remaining feature improves the score (by more than
         `min_improvement`), or once `max_features` (the target size N') is
         reached.

    Because it scores subsets with the model itself, FFS captures feature
    *interactions* that filter methods (correlation / Top-K / ANOVA F) miss --
    at the cost of being computationally expensive (many model fits) and
    dependent on the chosen classifier and its hyperparameters.

    Parameters:
    -----------
    X_train : np.ndarray
        Scaled training features (n_samples, n_features).
    y_train : np.ndarray
        Training labels.
    feature_names : list of str
        Names of the columns in X_train (used for reporting).
    n_estimators, learning_rate, random_state : 
        Hyperparameters for the wrapper XGBoost estimator.
    max_features : int or None
        Target number of features N'. None means keep adding until no feature
        improves the cross-validated score.
    cv : int
        Number of stratified cross-validation folds used to score each subset.
    scoring : str
        sklearn scoring metric used to compare subsets (default: 'accuracy').
    min_improvement : float
        Minimum gain in the CV score required to accept a new feature.

    Returns:
    --------
    tuple
        (selected_indices, selected_names, history)
        - selected_indices : column indices into X_train, in the order added
        - selected_names    : the corresponding feature names
        - history           : per-step log [{'feature', 'index', 'cv_score'}]
    """
    n_features = X_train.shape[1]
    target = n_features if max_features is None else min(max_features, n_features)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    selected: list[int] = [feature_names.index(feature) for feature in initial_features]
    remaining: list[int] = [i for i in range(n_features) if i not in selected] #list(range(n_features))
    history: list[dict[str, object]] = []
    best_score = -np.inf

    if verbose:
        print(f"\nForward Feature Selection (wrapper, {cv}-fold CV, scoring='{scoring}')")
        print(f"Candidate features: {n_features} | target N': {target}")

    while remaining and len(selected) < target:
        round_best_score = -np.inf
        round_best_feat: int | None = None

        # Try adding each remaining feature on top of the current set.
        for feat in remaining:
            trial = selected + [feat]
            model = build_model(**model_params, random_state = random_state)
            scores = cross_val_score(
                model, X_train[:, trial], y_train,
                cv=cv_splitter, scoring=scoring, n_jobs=5,
            )
            mean_score = float(scores.mean())
            if mean_score > round_best_score:
                round_best_score = mean_score
                round_best_feat = feat

        # Stop if the best candidate this round doesn't improve enough.
        if round_best_feat is None or round_best_score <= best_score + min_improvement:
            if verbose:
                print(f"  No further improvement (best candidate "
                      f"{round_best_score:.4f} vs current {best_score:.4f}). Stopping.")
            break

        selected.append(round_best_feat)
        remaining.remove(round_best_feat)
        best_score = round_best_score
        history.append({
            'feature': feature_names[round_best_feat],
            'index': round_best_feat,
            'cv_score': round_best_score,
        })
        if verbose:
            print(f" [ {len(selected)} ] + {feature_names[round_best_feat]:<25} "
                  f"CV {scoring}: {round_best_score:.4f}")

    selected_names = [feature_names[i] for i in selected]
    if verbose:
        print(f"Selected {len(selected)}/{n_features} features: {selected_names}")

    return selected, selected_names, history


def backward_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    model_params: dict[str, Any], 
    build_model: Callable,  
    random_state: int = 42,
    min_features: int = 1,
    cv: int = 5,
    scoring: str = 'accuracy',
    min_improvement: float = 1e-6,
    verbose: bool = True,
) -> tuple[list[int], list[str], list[dict[str, object]]]:
    """
    Backward Feature Selection (BFS) -- a *wrapper* method.

    Greedy procedure described in the course notes:
      1. Start with the FULL feature set.
      2. At each step, try removing each currently selected feature and score the
         resulting subset using the actual classifier, evaluated with
         cross-validation.
      3. Permanently remove the single feature whose removal improves the score
         the most (or least hurts the score).
      4. Stop when no remaining feature removal improves the score (by more than
         `min_improvement`), or once `min_features` (the target size N') is
         reached.

    Because it scores subsets with the model itself, BFS captures feature
    *interactions* that filter methods miss -- at the cost of being computationally
    expensive (many model fits) and dependent on the chosen classifier and its
    hyperparameters.

    Parameters:
    -----------
    X_train : np.ndarray
        Scaled training features (n_samples, n_features).
    y_train : np.ndarray
        Training labels.
    feature_names : list of str
        Names of the columns in X_train (used for reporting).
    n_estimators, learning_rate, random_state : 
        Hyperparameters for the wrapper XGBoost estimator.
    min_features : int
        Target number of features N' (minimum to keep). Stop when we reach this
        or when no further removal improves the score.
    cv : int
        Number of stratified cross-validation folds used to score each subset.
    scoring : str
        sklearn scoring metric used to compare subsets (default: 'accuracy').
    min_improvement : float
        Minimum gain in the CV score required to accept a removal.

    Returns:
    --------
    tuple
        (selected_indices, selected_names, history)
        - selected_indices : column indices into X_train, in order of removal
        - selected_names    : the corresponding feature names remaining
        - history           : per-step log [{'feature', 'index', 'cv_score'}]
    """
    n_features = X_train.shape[1]
    target = max(min_features, 1)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    selected: list[int] = list(range(n_features))
    history: list[dict[str, object]] = []
    
    # Compute baseline score with all features
    model = build_model(**model_params, random_state = random_state)
    scores = cross_val_score(
        model, X_train[:, selected], y_train,
        cv=cv_splitter, scoring=scoring, n_jobs=5,
    )
    best_score = float(scores.mean())

    if verbose:
        print(f"\nBackward Feature Selection (wrapper, {cv}-fold CV, scoring='{scoring}')")
        print(f"Starting features: {n_features} | target N': {target}")
        print(f"  [Baseline with all {n_features} features] CV {scoring}: {best_score:.4f}")

    step = 0
    while len(selected) > target:
        round_best_score = -np.inf
        round_best_feat_to_remove: int | None = None

        # Try removing each currently selected feature one at a time.
        for feat in selected:
            trial = [f for f in selected if f != feat]
            
            # Skip if removing this feature would leave us with no features
            if not trial:
                continue
            
            model = build_model(**model_params, random_state = random_state)
            scores = cross_val_score(
                model, X_train[:, trial], y_train,
                cv=cv_splitter, scoring=scoring, n_jobs=5,
            )
            mean_score = float(scores.mean())
            if mean_score > round_best_score:
                round_best_score = mean_score
                round_best_feat_to_remove = feat

        # Stop if the best removal this round doesn't improve enough.
        if round_best_feat_to_remove is None or round_best_score <= best_score + min_improvement:
            if verbose:
                print(f"  No further improvement (best removal "
                      f"{round_best_score:.4f} vs current {best_score:.4f}). Stopping.")
            break

        selected.remove(round_best_feat_to_remove)
        best_score = round_best_score
        step += 1
        history.append({
            'feature': feature_names[round_best_feat_to_remove],
            'index': round_best_feat_to_remove,
            'cv_score': round_best_score,
        })
        if verbose:
            print(f"  [{step:>2}] - {feature_names[round_best_feat_to_remove]:<25} "
                  f"CV {scoring}: {round_best_score:.4f}")

    selected_names = [feature_names[i] for i in sorted(selected)]
    if verbose:
        print(f"Selected {len(selected)}/{n_features} features: {selected_names}")

    return sorted(selected), selected_names, history



def train_classifier(
    model: xgb.XGBClassifier | RandomForestClassifier,
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 100, 
    model_type: str = "XGBoost",
) -> xgb.XGBClassifier | RandomForestClassifier:
    """
    Train an XGBoost classifier for 5-class classification.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Scaled training features
    y_train : np.ndarray
        Training labels (0-5)
    n_estimators : int
        Number of boosting rounds (default: 100)
    random_state : int
        Random seed for reproducibility
    learning_rate : float
        Learning rate (eta) for boosting (default: 0.1)
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained XGBoost model
    """
    
    model.fit(X_train, y_train)
    print(f"{model_type} model trained successfully with {n_estimators} estimators")
    
    return model



def tune_hyperparameters(
    model: xgb.XGBClassifier | RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list] | None = None,
    cv: int = 5,
    scoring: str = 'accuracy',
    random_state: int = 42,
    #num_class: int | None = None,
    verbose: bool = True,
) -> tuple[dict[str, int | float], float]:
    """
    Tune model (XGBoost or RandomForest) with cross-validated grid search.

    IMPORTANT: this must be run on the TRAINING data only. The held-out test
    set is never seen during tuning, so the final test evaluation stays an
    unbiased estimate of generalization.

    Each candidate hyperparameter combination is scored with Stratified k-Fold
    cross-validation (the same validation scheme used elsewhere in the
    pipeline); the combination with the best mean CV score is returned.

    Parameters:
    -----------
    model : xgb.XGBClassifier | RandomForestClassifier
        The model to tune ( either XGBoost or RandomForest)
    X : np.ndarray
        Scaled training features. If feature selection ran first, pass the
        already-reduced matrix so tuning matches the final feature space.
    y : np.ndarray
        Training labels.
    param_grid : dict or None
        {param_name: [values, ...]} to search. None uses a sensible default
        grid over n_estimators, max_depth, learning_rate, and subsample.
    cv : int
        Number of stratified folds used to score each combination.
    scoring : str
        sklearn scoring metric to optimize (default: 'accuracy').
    num_class : int or None
        Passed to the base estimator. None lets XGBoost infer per fold (robust
        when a fold is missing a rare class).

    Returns:
    --------
    tuple
        (best_params, best_cv_score)
        best_params : the winning hyperparameter dict (feed into _build_xgb /
                      train_classifier for the final fit)
        best_cv_score : mean CV score of the best combination
    """
    if param_grid is None:
        param_grid = {
            'n_estimators':  [100, 200, 300, 400],
            'max_depth':     [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'subsample':     [0.4, 0.55, 0.6, 0.7, 1.0],
        }

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    #base = build_xgb(random_state=random_state, num_class=num_class)

    if verbose:
        n_combos = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"\n{'='*60}")
        print(f"Hyperparameter Tuning (grid search, {cv}-fold CV, "
              f"scoring='{scoring}')")
        print(f"Searching {n_combos} combinations x {cv} folds "
              f"= {n_combos * cv} fits")
        print(f"{'='*60}")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_splitter,
        n_jobs=-1,
        refit=False,  # we refit the final model ourselves downstream
    )
    search.fit(X, y)

    best_params = dict(search.best_params_)
    best_score = float(search.best_score_)

    if verbose:
        print(f"Best CV {scoring}: {best_score:.4f}")
        print(f"Best params: {best_params}")
        print(f"{'='*60}")

    return best_params, best_score


