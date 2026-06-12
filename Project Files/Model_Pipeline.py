from typing import Any, Callable
import numpy as np
from Data_Loader import load_data, preprocess_data, encode_target_to_classes, split_data, split_data_by_cup, split_data_with_indices, split_data_by_cup_with_indices, scale_features, anova_f_scores
from Flow_Rate import compute_flow_rates
from Model_Builder import forward_feature_selection, train_classifier, backward_feature_selection, tune_hyperparameters, build_xgb, build_randomforest
from Model_Test import evaluate_model, get_feature_importance, get_combined_prediction_results
from Model_Validation import crosss_val_model, cross_validate_model
from Model_Saving import save_model, load_model
from Graph_results import plot_anova_f_scores, graph_CM, graph_flow_rate, graph_cv_comparison, plot_feature_impact

print("hello there")

def handle_data(
    csv_path: str, 
    target_column: str,
    split_by_cup: bool = False,
    cup_number: int =1,
):
    # Load data
    data = load_data(csv_path)
        
    # Preprocess data
    X, y, cup_labels, features = preprocess_data(data, target_column)
    y_encoded, encoder = encode_target_to_classes(y)
    
    # F-score of features
    anova_results = anova_f_scores(X,y_encoded)
    # plot_anova_f_scores(anova_results, top_n=20)

    #Split to train,test (index-aware so test rows can be mapped back to the CSV)
    if not split_by_cup:
        X_train, X_test, y_train, y_test, train_idx, test_idx= split_data_with_indices(X, y_encoded, test_size=0.2)
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx= split_data_by_cup_with_indices(X, y_encoded, cup_labels, cup_number=cup_number)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, scaler, y_train, y_test, encoder, features, anova_results, test_idx


def model_pipeline(
    # data
    X_train_scaled, X_test_scaled, scaler, y_train, y_test, encoder, features, anova_results,
    # model config
    model_params: dict[str, Any], build_model: Callable, random_state: int = 42, model_type: str = "model",
    # feature selection
    use_feature_selection: bool = True, max_features: int| None = None, n_initial: int = 10,
    # cross validation
    run_cross_validation = True, cv: int = 5,
    # feature turning
    tune: bool = True, tune_grid: dict[str, list] | None = None, tune_cv: int = 5,
    #
    feature_impact_graph: bool = False
) -> dict[str, Any]:

    model_options = model_params.copy() # copy of model_params to keep original the same after hyperparameter tunning
    
    # Forward feature selection
    selected_indices: list[int] | None = None
    selected_features = features
    if use_feature_selection:
        selected_indices, selected_features, _ = forward_feature_selection(
            X_train_scaled,
            y_train,
            features,
            model_params,
            build_model,
            max_features= max_features,
            cv=cv,
            initial_features= anova_results['feature'].head(n_initial).tolist()
        )
        X_train_scaled = X_train_scaled[:, selected_indices]
        X_test_scaled = X_test_scaled[:, selected_indices]

    model = build_model(**model_params, random_state= random_state)
    
    if feature_impact_graph:
        plot_feature_impact(
        np.concatenate([X_train_scaled, X_test_scaled], axis=0),
        np.concatenate([y_train, y_test], axis=0),
        model,
        anova_results['feature'].tolist(), features)

    best_params: dict[str, int | float] | None = None
    best_tune_score: float | None = None
    if tune:
        best_params, best_tune_score = tune_hyperparameters(
            model,
            X_train_scaled,
            y_train,
            param_grid=tune_grid,
            cv=tune_cv,
        )
        model_options.update(best_params)
        model = build_model(**model_options, random_state= random_state)
    
    # K-Fold cross-validation on the (selected) training data for a robust
    # estimate of generalization performance before committing to final fit.
    cv_results: dict[str, object] | None = None
    if run_cross_validation:
        cv_results = cross_validate_model(
            model,
            np.concatenate([X_train_scaled, X_test_scaled], axis=0),
            np.concatenate([y_train, y_test], axis=0),
            cv=cv,
        )

    # Train XGBoost model
    model = train_classifier(
        model,
        X_train_scaled, 
        y_train
    )

    # Evaluate
    results = evaluate_model(model, X_test_scaled, y_test, encoder)
    
    # Feature importance (over the selected features only)
    get_feature_importance(model, selected_features)
    
    #crosss_val_model(X, y_encoded, cup_labels, n_estimators, learning_rate, cv=3)
    
    # Save model
    #save_model(model, scaler, encoder, selected_indices=selected_indices, model_type= model_type)
    
    return {
        'scaler': scaler,
        'encoder': encoder,
        'features': features,
        'model': model,
        'accuracy': results['accuracy'],
        'confusion_matrix': results['confusion_matrix'],
        'probablities': results['probablities'],
        'cv_results': cv_results
    }



CSV_PATH = r'cup_dataset.csv'

X_train_scaled, X_test_scaled, scaler, y_train, y_test, encoder, features, anova_results, test_idx = handle_data(
    CSV_PATH,
    target_column='Fill', 
    split_by_cup= False, cup_number=1
    )

# XGBoost model options
xgb_options = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.4,
    'colsample_bytree': 0.55,
    'objective': 'multi:softprob',
    'n_jobs': 5,
    'verbosity': 0,
    'reg_alpha': 0.5,
    'reg_lambda': 0.3
}

xgb_tune_grid = {
            'n_estimators':  [200, 300, 400],
            'max_depth':     [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'reg_alpha': [0.3, 0.5, 0.7],
            'reg_lambda': [0.1, 0.3, 0.5]
        }

print('='*60)
print("XGBoost")
print('='*60)
results_XGB = model_pipeline(
    X_train_scaled, X_test_scaled, scaler, y_train, y_test, encoder, features, anova_results,
    xgb_options, build_xgb, model_type= "XGB",
    random_state= 42,
    use_feature_selection= True, max_features=100, n_initial=30,
    run_cross_validation = True, cv = 4,
    tune = True, tune_grid= xgb_tune_grid, tune_cv= 4,
    feature_impact_graph= False
    )
# best results: Tune off, FFS on (10 initials) 
# graph_CM(results_XGB['confusion_matrix'], encoder.classes_.astype('str'))

# forest_options = {
#     "n_estimators": 200,
#     #"criterion": "gini",
#     "max_depth": 8,
#     "min_samples_split": 2,
#     "min_samples_leaf": 1,
#     #"min_weight_fraction_leaf": 0.0,
#     "max_features": "log2",
#    # "max_leaf_nodes": None,
#    # "min_impurity_decrease": 0.0,
#     #"bootstrap": True,
#     #"oob_score": False,
#     "n_jobs": 3,
#     "verbosity": 0,
#    # "warm_start": False,
#     "class_weight": None,
#     #"ccp_alpha": 0.0,
#    # "max_samples": None,
# }

# forest_tune_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth':    [6, 8, 10,],
#     'max_features': ['sqrt', 'log2'],
# }

# print('\n')
# print('='*60)
# print("Random Forest")
# print('='*60)
# results_Forest = model_pipeline(
#     X_train_scaled, X_test_scaled, scaler, y_train, y_test, encoder, features, anova_results,
#     forest_options, build_randomforest, model_type="FOREST",
#     random_state= 42,
#     use_feature_selection= True, max_features=100, n_initial= 25,
#     run_cross_validation = True, cv = 4,
#     tune = False, tune_grid= forest_tune_grid, tune_cv= 4,
#     feature_impact_graph= False
#     )

# graph_CM(results_Forest['confusion_matrix'], encoder.classes_.astype('str'))

# print('\n')
# print('='*60)
# print("Combined")
# print('='*60)
# XGB_CONST = 0.5
# FORSET_CONST = 0.5
# final_results = get_combined_prediction_results(
#     XGB_CONST, results_XGB['probablities'], 
#     FORSET_CONST, results_Forest['probablities'], 
#     y_test, encoder)

# graph_cv_comparison(
#     {
#         'XGB': results_XGB.get('cv_results'),
#         'Forest': results_Forest.get('cv_results'),
#     },
#     cv_label="Stratified 4-Fold CV",
# )


# --- Flow rate by REGRESSION: train + tune XGB and RandomForest regressors on
#     the same test split, pick whichever model+hyperparameters fit best (CV),
#     and predict the flow rate directly. (No longer derived from the fill
#     classifier or a formula.) ---
# flow_results = compute_flow_rates(
#     CSV_PATH,
#     test_indices=test_idx,
#     tune=True,
#     cv=4,
#     random_state=42,
# )
# print(f"Flow-rate model chosen: {flow_results['best_model']} "
#       f"(params={flow_results['best_params']}, "
#       f"test R2={flow_results['metrics']['R2']:.4f})")

# # Flow rate figures: bars (true vs predicted by fill class) + per-sample scatter.
# graph_flow_rate(flow_results)