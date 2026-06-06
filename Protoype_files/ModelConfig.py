from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from Data_Loader import load_data, preprocess_data, encode_target_to_classes, split_data, split_data_by_cup, scale_features
from sklearn.model_selection import StratifiedKFold, cross_validate

data = load_data('cup_dataset.csv')
X, y, cup_labels, features = preprocess_data(data, 'Fill')
y_encoded, encoder = encode_target_to_classes(y)
X_train, X_test, y_train, y_test = split_data(X, y_encoded, test_size=0.25)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

Ridge = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 1], cv=4, class_weight='balanced').fit(X_train_scaled, y_train)
print("Ridge: ", Ridge.score(X_test_scaled, y_test))

Logic = LogisticRegressionCV(
    cv=3, random_state=42,
    l1_ratios=(0,),
    scoring="accuracy",
    max_iter=5000
).fit(X_train_scaled, y_train)

print("Logistic: ", Logic.score(X_test_scaled, y_test))


Forest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=3)
Forest.fit(X_train_scaled, y_train)
print("Forest: ", Forest.score(X_test_scaled, y_test))

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

# Usage:
print('='*60)
print('Ridge')
results, raw_results = kfold_cv_multiple(Ridge, X, y, k=5)

for metric, values in results.items():
    print(f"{metric.upper()}: {values['mean']:.4f} (+/- {values['std']:.4f})")

print('='*60)

print('Logic')
results, raw_results = kfold_cv_multiple(Logic, X, y, k=5)

for metric, values in results.items():
    print(f"{metric.upper()}: {values['mean']:.4f} (+/- {values['std']:.4f})")

print('='*60)

print('Forest')
results, raw_results = kfold_cv_multiple(Forest, X, y, k=5)

for metric, values in results.items():
    print(f"{metric.upper()}: {values['mean']:.4f} (+/- {values['std']:.4f})")

print('='*60)