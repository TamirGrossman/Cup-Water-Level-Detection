import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import pickle


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    data = pd.read_csv(csv_path)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def preprocess_data(
    data: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """
    Preprocess data: handle missing values, separate features and target, 
    and split into train/test sets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw dataset
    target_column : str
        Name of the target column (classification labels)
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, label_encoder)
    """
    # Handle missing values
    data = data.dropna()
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Encode categorical features if needed
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        feature_names = X.columns.tolist()
    
    # Encode target labels (convert to 0-5 for 6 classes)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {np.unique(y_encoded)}")
    
    return X_train, X_test, y_train, y_test, feature_names, label_encoder


def scale_features(
    X_train: pd.DataFrame | np.ndarray, 
    X_test: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def _build_xgb(
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    random_state: int = 42,
    num_class: int | None = 15,
    verbosity: int = 0,
    max_depth: int = 6,
    subsample: float = 0.7,
    colsample_bytree: float = 0.7,
) -> xgb.XGBClassifier:
    """
    Construct an (untrained) XGBoost classifier with the project's
    default hyperparameters.

    Used both for final training and inside forward feature selection so the
    "wrapper" estimator that scores feature subsets matches the model that is
    ultimately deployed.

    Parameters:
    -----------
    num_class : int or None
        Accepted for backward compatibility but NO LONGER passed to XGBoost.
        The scikit-learn wrapper (XGBClassifier) infers the number of classes
        from the labels at fit time and sets both the objective and num_class
        itself. Passing num_class explicitly alongside objective='multi:softprob'
        makes the wrapper conflict with itself, which crashes inside parallel CV
        workers (the cryptic joblib traceback). So we deliberately ignore it.
    max_depth, subsample, colsample_bytree :
        Tunable hyperparameters. Defaults preserve the original fixed values, so
        existing callers behave identically; the tuner overrides them.
    """
    # Do NOT set `objective` or `num_class` manually. XGBClassifier detects the
    # multiclass case from y and configures multi:softprob + num_class on its
    # own. Setting them by hand is what produced the conflict/crash.
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        # Keep the estimator single-threaded. It is almost always run inside an
        # outer parallel loop (cross_val_score / GridSearchCV with n_jobs=-1).
        # Letting each estimator also spawn threads creates nested parallelism,
        # which on Windows can exhaust resources or deadlock and surface as a
        # cryptic joblib crash. The outer CV provides the parallelism instead.
        n_jobs=1,
        verbosity=verbosity,
    )
    return xgb.XGBClassifier(**params)


def forward_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
    max_features: int | None = None,
    cv: int = 5,
    scoring: str = 'accuracy',
    min_improvement: float = 1e-4,
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

    selected: list[int] = []
    remaining: list[int] = list(range(n_features))
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
            model = _build_xgb(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                num_class=None,  # let XGBoost infer per fold
            )
            scores = cross_val_score(
                model, X_train[:, trial], y_train,
                cv=cv_splitter, scoring=scoring, n_jobs=-1,
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
            print(f"  [{len(selected):>2}] + {feature_names[round_best_feat]:<25} "
                  f"CV {scoring}: {round_best_score:.4f}")

    selected_names = [feature_names[i] for i in selected]
    if verbose:
        print(f"Selected {len(selected)}/{n_features} features: {selected_names}")

    return selected, selected_names, history


def train_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 100, 
    random_state: int = 42, 
    learning_rate: float = 0.1,
    best_params: dict[str, object] | None = None
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier for 6-class classification.
    
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
    best_params : dict or None
        Tuned hyperparameters from tune_hyperparameters(). When provided, these
        override n_estimators / learning_rate and any other tuned keys
        (e.g. max_depth, subsample). None keeps the passed-in defaults.
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained XGBoost model
    """
    # Start from the explicit args, then let tuned params override them.
    # Note: num_class/objective are intentionally NOT set here -- the XGBoost
    # sklearn wrapper infers them from y at fit time.
    build_kwargs: dict[str, object] = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=1,
    )
    if best_params:
        build_kwargs.update(best_params)

    model = _build_xgb(**build_kwargs)
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
        objective='multi:softprob', # 6-class multiclass
        num_class=6,                # Number of classes
        random_state=42
    )
    '''
    model.fit(X_train, y_train, verbose=False)
    eff_estimators = model.get_params().get('n_estimators', n_estimators)
    print(f"XGBoost model trained successfully with {eff_estimators} estimators")
    
    return model


def evaluate_model(
    model: xgb.XGBClassifier, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    label_encoder: LabelEncoder
) -> dict[str, object]: 
    """
    Evaluate the trained model and print detailed metrics.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained classification model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    label_encoder : LabelEncoder
        Encoder to convert numeric labels back to original class names
        
    Returns:
    --------
    dict
        Dictionary with accuracy and predictions
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    
    # Convert numeric labels back to original class names
    class_names = label_encoder.classes_.astype('str')

    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ── Per-class accuracy and recall ─────────────────────────────────────────
    # Derived directly from the confusion matrix (rows = true, cols = predicted).
    #   recall_i   = TP_i / (all true samples of class i)        = cm[i,i] / row_i
    #   accuracy_i = (TP_i + TN_i) / total  (one-vs-rest: how often the model is
    #                correct about "is this class i or not", counting correct
    #                rejections of other classes too)
    n_classes = cm.shape[0]
    total = cm.sum()
    row_totals = cm.sum(axis=1)   # true count per class
    col_totals = cm.sum(axis=0)   # predicted count per class
    tp = np.diag(cm)

    per_class_recall = np.divide(
        tp, row_totals,
        out=np.zeros(n_classes, dtype=float), where=row_totals != 0
    )
    # One-vs-rest accuracy: TP + TN over total. TN = total - row_i - col_i + TP_i.
    tn = total - row_totals - col_totals + tp
    per_class_accuracy = (tp + tn) / total if total > 0 else np.zeros(n_classes)

    print("\nPer-Class Metrics:")
    print(f"{'Class':<20}{'Accuracy':>12}{'Recall':>12}{'Support':>10}")
    for i in range(n_classes):
        print(f"{class_names[i]:<20}{per_class_accuracy[i]:>12.4f}"
              f"{per_class_recall[i]:>12.4f}{int(row_totals[i]):>10}")

    per_class = {
        class_names[i]: {
            'accuracy': float(per_class_accuracy[i]),
            'recall':   float(per_class_recall[i]),
            'support':  int(row_totals[i]),
        }
        for i in range(n_classes)
    }

    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'per_class': per_class,
        'per_class_accuracy': per_class_accuracy,
        'per_class_recall': per_class_recall,
    }


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list] | None = None,
    cv: int = 5,
    scoring: str = 'accuracy',
    random_state: int = 42,
    num_class: int | None = None,
    verbose: bool = True,
) -> tuple[dict[str, object], float]:
    """
    Tune XGBoost hyperparameters with cross-validated grid search.

    IMPORTANT: this must be run on the TRAINING data only. The held-out test
    set is never seen during tuning, so the final test evaluation stays an
    unbiased estimate of generalization.

    Each candidate hyperparameter combination is scored with Stratified k-Fold
    cross-validation (the same validation scheme used elsewhere in the
    pipeline); the combination with the best mean CV score is returned.

    Parameters:
    -----------
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
            'n_estimators':  [100, 200, 300],
            'max_depth':     [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample':     [0.7, 1.0],
        }

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    base = _build_xgb(random_state=random_state, num_class=num_class)

    if verbose:
        n_combos = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"\n{'='*60}")
        print(f"Hyperparameter Tuning (grid search, {cv}-fold CV, "
              f"scoring='{scoring}')")
        print(f"Searching {n_combos} combinations x {cv} folds "
              f"= {n_combos * cv} fits")
        print(f"{'='*60}")

    search = GridSearchCV(
        estimator=base,
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


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
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

    results: dict[str, object] = {}
    for metric in scoring:
        # Rebuild a fresh estimator for each metric; num_class=None lets XGBoost
        # infer the class count per fold (robust to rare classes missing).
        model = _build_xgb(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            num_class=None,
        )
        scores = cross_val_score(
            model, X, y, cv=cv_splitter, scoring=metric, n_jobs=-1,
        )
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


def get_feature_importance(
    model: xgb.XGBClassifier, 
    feature_names: list[str], 
    top_n: int = 10
) -> None:
    """
    Extract and display the most important features from XGBoost model.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained XGBoost model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display (default: 10)
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    print(f"\nTop {top_n} Important Features (Gain-based):")
    for rank, idx in enumerate(indices, 1):
        print(f"{rank}. {feature_names[idx]}: {importances[idx]:.4f}")


def predict_single(
    model: xgb.XGBClassifier, 
    features: np.ndarray, 
    scaler: StandardScaler, 
    label_encoder: LabelEncoder,
    selected_indices: list[int] | None = None
) -> tuple[str, float]:
    """
    Make prediction on a single sample.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    features : np.ndarray or list
        Input features (must match the FULL training data structure, i.e. all
        features the scaler was fitted on -- subsetting happens after scaling)
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
    selected_indices : list[int] or None
        Column indices chosen by forward feature selection. If provided, the
        scaled features are reduced to this subset before prediction. None uses
        all features.
        
    Returns:
    --------
    tuple
        (predicted_class_name, confidence_score)
    """
    features_scaled = scaler.transform([features])
    if selected_indices is not None:
        features_scaled = features_scaled[:, selected_indices]
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = np.max(probabilities)
    
    class_name = label_encoder.inverse_transform([prediction])[0]
    
    return class_name, confidence


def predict_batch(
    model: xgb.XGBClassifier, 
    features: np.ndarray | list[list[float]], 
    scaler: StandardScaler, 
    label_encoder: LabelEncoder,
    selected_indices: list[int] | None = None
) -> pd.DataFrame:
    """
    Make predictions on multiple samples and return results with confidence scores.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    features : np.ndarray or list of lists
        Input features for multiple samples (full feature structure)
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
    selected_indices : list[int] or None
        Column indices chosen by forward feature selection. If provided, the
        scaled features are reduced to this subset before prediction.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions and confidence scores
    """
    features_scaled = scaler.transform(features)
    if selected_indices is not None:
        features_scaled = features_scaled[:, selected_indices]
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    confidences = np.max(probabilities, axis=1)
    
    class_names = label_encoder.inverse_transform(predictions)
    
    return pd.DataFrame({
        'predicted_class': class_names,
        'confidence': confidences
    })


def save_model(
    model: xgb.XGBClassifier, 
    scaler: StandardScaler, 
    label_encoder: LabelEncoder, 
    model_path: str = 'xgboost_model.pkl',
    selected_indices: list[int] | None = None
) -> None:
    """
    Save trained XGBoost model and preprocessing objects.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
    model_path : str
        Path to save the model
    selected_indices : list[int] or None
        Feature indices chosen by forward feature selection (if any)
    """
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'encoder': label_encoder,
            'selected_indices': selected_indices,
        }, f)
    print(f"Model saved to {model_path}")


def load_model(
    model_path: str = 'xgboost_model.pkl'
) -> tuple[xgb.XGBClassifier, StandardScaler, LabelEncoder, list[int] | None]:
    """
    Load a saved XGBoost model and preprocessing objects.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    tuple
        (model, scaler, label_encoder, selected_indices)
    """
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['encoder'], data.get('selected_indices')


def train_classification_pipeline(
    csv_path: str, 
    target_column: str, 
    n_estimators: int = 100, 
    learning_rate: float = 0.1,
    use_feature_selection: bool = True,
    max_features: int | None = None,
    cv: int = 5,
    run_cross_validation: bool = True,
    cv_folds: int = 5,
    tune: bool = True,
    tune_grid: dict[str, list] | None = None,
    tune_cv: int = 5
) -> dict[str, object]:
    """
    Complete pipeline to train and evaluate a 6-class XGBoost classifier.
    
    Parameters:
    -----------
    csv_path : str
        Path to training CSV file
    target_column : str
        Name of the target column in CSV
    n_estimators : int
        Number of boosting rounds (default: 100)
    learning_rate : float
        Learning rate for XGBoost (default: 0.1)
    use_feature_selection : bool
        If True, run forward feature selection (wrapper) before final training
    max_features : int or None
        Target number of features N' for forward selection (None = no cap)
    cv : int
        Cross-validation folds used to score subsets during feature selection
    run_cross_validation : bool
        If True, run Stratified k-Fold cross-validation on the (selected)
        training data before final training, for a robust performance estimate
    cv_folds : int
        Number of folds k for the cross-validation evaluation (default: 5)
    tune : bool
        If True, run cross-validated grid-search hyperparameter tuning on the
        training data (after feature selection) before the final fit. The tuned
        params are then used for both the k-fold report and final training.
    tune_grid : dict or None
        Hyperparameter grid for tuning. None uses the default grid in
        tune_hyperparameters().
    tune_cv : int
        Number of stratified folds used during hyperparameter tuning.
        
    Returns:
    --------
    dict
        Dictionary containing model, scaler, encoder, and results
    """
    # Load and preprocess
    data = load_data(csv_path)
    X_train, X_test, y_train, y_test, features, encoder = preprocess_data(
        data, target_column
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Forward feature selection (wrapper around the XGBoost classifier)
    selected_indices: list[int] | None = None
    selected_features = features
    if use_feature_selection:
        selected_indices, selected_features, _ = forward_feature_selection(
            X_train_scaled,
            y_train,
            features,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_features=max_features,
            cv=cv,
        )
        X_train_scaled = X_train_scaled[:, selected_indices]
        X_test_scaled = X_test_scaled[:, selected_indices]
    
    # Hyperparameter tuning via cross-validated grid search, on the TRAINING
    # data only (after feature selection so it matches the final feature space).
    # The held-out test set is never touched here, so final test metrics stay
    # unbiased. Tuned params feed both the k-fold report and the final fit.
    best_params: dict[str, object] | None = None
    best_tune_score: float | None = None
    if tune:
        best_params, best_tune_score = tune_hyperparameters(
            X_train_scaled,
            y_train,
            param_grid=tune_grid,
            cv=tune_cv,
        )

    # Effective hyperparameters used downstream (tuned if available).
    eff_n_estimators = best_params.get('n_estimators', n_estimators) if best_params else n_estimators
    eff_learning_rate = best_params.get('learning_rate', learning_rate) if best_params else learning_rate

    # K-Fold cross-validation on the (selected) training data for a robust
    # estimate of generalization performance before committing to final fit.
    cv_results: dict[str, object] | None = None
    if run_cross_validation:
        cv_results = cross_validate_model(
            X_train_scaled,
            y_train,
            n_estimators=eff_n_estimators,
            learning_rate=eff_learning_rate,
            cv=cv_folds,
        )
    
    # Train XGBoost model (using tuned hyperparameters if tuning ran)
    model = train_classifier(
        X_train_scaled, 
        y_train, 
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        best_params=best_params
    )
    
    # Evaluate
    results = evaluate_model(model, X_test_scaled, y_test, encoder)
    
    # Feature importance (over the selected features only)
    get_feature_importance(model, selected_features)
    
    # Save model
    save_model(model, scaler, encoder, selected_indices=selected_indices)
    
    return {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'features': features,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'results': results,
        'cv_results': cv_results,
        'best_params': best_params,
        'best_tune_score': best_tune_score
    }


# Train the XGBoost model
#
# IMPORTANT (Windows): the __name__ == '__main__' guard is REQUIRED. joblib
# (used by cross_val_score / GridSearchCV with n_jobs=-1) spawns worker
# processes on Windows by re-importing this module. Without the guard, each
# worker would re-run train_classification_pipeline on import, recursively
# spawning more workers -- another common source of the cryptic joblib crash.
def main() -> None:
    pipeline = train_classification_pipeline(
        'cup_dataset.csv',
        target_column='Fill',
        n_estimators=200,
        learning_rate=0.05,
        use_feature_selection=True,   # run forward feature selection first
        max_features=None,            # set e.g. 8 to cap the selected subset size (N')
        run_cross_validation=True,    # run Stratified k-Fold CV before final training
        cv_folds=5,                   # number of folds k
        tune=True,                    # tune hyperparameters (CV grid search) before testing
        tune_cv=5                     # folds used during tuning
    )

    # Single prediction
    # NOTE: pass the FULL feature vector (all columns the scaler saw); the pipeline
    # reduces it to the selected features internally via selected_indices.
    # new_sample = [5.1, 3.5, 1.4, 0.2, 0]
    # prediction, confidence = predict_single(
    #     pipeline['model'],
    #     new_sample,
    #     pipeline['scaler'],
    #     pipeline['encoder'],
    #     selected_indices=pipeline['selected_indices']
    # )
    # print(f"Predicted class: {prediction} (confidence: {confidence:.2%})")

    # Batch predictions
    # batch_samples = [[5.1, 3.5, 1.4, 0.2, 0], [6.2, 2.9, 4.3, 1.3, 1]]
    # results = predict_batch(
    #     pipeline['model'],
    #     batch_samples,
    #     pipeline['scaler'],
    #     pipeline['encoder'],
    #     selected_indices=pipeline['selected_indices']
    # )
    # print(results)

    # Load model later
    model, scaler, encoder, selected_indices = load_model()


if __name__ == '__main__':
    main()