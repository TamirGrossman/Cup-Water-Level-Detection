import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(
    model: xgb.XGBClassifier | RandomForestClassifier, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    label_encoder: LabelEncoder
) -> dict[str, object]: 
    """
    Evaluate the trained model and print detailed metrics.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier or RandomForestClassifier
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
        Dictionary with accuracy and predictions:
        {
        'accuracy': accuracy -> model accuracy on test
        'predictions': y_pred -> model predictions
        'confusion_matrix': cm -> model confusion matrix
        'probablities': y_probs -> model probabilites for each class
        }
    """
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    # Convert numeric labels back to original class names
    class_names = label_encoder.classes_.astype('str')

    accuracy, cm = repot_on_results(y_test, y_pred, class_names)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'probablities': y_probs
    }

def get_combined_prediction_results(
        XGB_CONST:float, probs_XGB: np.ndarray, 
        FOREST_CONST:float, probs_Forest: np.ndarray, 
        y_test: np.ndarray,
        label_encoder: LabelEncoder
        )-> dict[str, object]:
    """
    Combines both model probabilities by weighted sum and evaluating
    """
    probs = XGB_CONST * probs_XGB+ FOREST_CONST * probs_Forest
    results = probs.argmax(axis = 1)

    # Convert numeric labels back to original class names
    class_names = label_encoder.classes_.astype('str')

    accuracy, cm = repot_on_results(y_test, results, class_names)

    return {
        'accuracy': accuracy,
        'predictions': results,
        'confusion_matrix': cm,
    }

def repot_on_results(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Helper function to create a final report on classification
    """

    accuracy = accuracy_score(y_true, y_pred)

    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return accuracy, cm

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


def get_feature_importance(
    model: xgb.XGBClassifier | RandomForestClassifier,
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

