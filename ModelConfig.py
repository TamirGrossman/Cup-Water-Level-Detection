import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    relevant_features: list[str] = [],
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
    if(len(relevant_features) > 0):
        X = X[relevant_features]

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


def train_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 100, 
    random_state: int = 42, 
    learning_rate: float = 0.1
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
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained XGBoost model
    """
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=6,
        random_state=random_state,
        n_jobs=-1,
        verbosity=1
    )
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
    print(f"XGBoost model trained successfully with {n_estimators} estimators")
    
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
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm
    }


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
    label_encoder: LabelEncoder
) -> tuple[str, float]:
    """
    Make prediction on a single sample.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    features : np.ndarray or list
        Input features (must match training data structure)
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
        
    Returns:
    --------
    tuple
        (predicted_class_name, confidence_score)
    """
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = np.max(probabilities)
    
    class_name = label_encoder.inverse_transform([prediction])[0]
    
    return class_name, confidence


def predict_batch(
    model: xgb.XGBClassifier, 
    features: np.ndarray | list[list[float]], 
    scaler: StandardScaler, 
    label_encoder: LabelEncoder
) -> pd.DataFrame:
    """
    Make predictions on multiple samples and return results with confidence scores.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    features : np.ndarray or list of lists
        Input features for multiple samples
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions and confidence scores
    """
    features_scaled = scaler.transform(features)
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
    model_path: str = 'xgboost_model.pkl'
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
    """
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'encoder': label_encoder}, f)
    print(f"Model saved to {model_path}")


def load_model(
    model_path: str = 'xgboost_model.pkl'
) -> tuple[xgb.XGBClassifier, StandardScaler, LabelEncoder]:
    """
    Load a saved XGBoost model and preprocessing objects.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    tuple
        (model, scaler, label_encoder)
    """
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['encoder']


def train_classification_pipeline(
    csv_path: str, 
    target_column: str,
    relevant_features: list[str] = [],
    n_estimators: int = 100, 
    learning_rate: float = 0.1
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
        
    Returns:
    --------
    dict
        Dictionary containing model, scaler, encoder, and results
    """
    # Load and preprocess
    data = load_data(csv_path)
    X_train, X_test, y_train, y_test, features, encoder = preprocess_data(
        data, target_column, relevant_features=relevant_features)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train XGBoost model
    model = train_classifier(
        X_train_scaled, 
        y_train, 
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    
    # Evaluate
    results = evaluate_model(model, X_test_scaled, y_test, encoder)
    
    # Feature importance
    get_feature_importance(model, features)
    
    # Save model
    save_model(model, scaler, encoder)
    
    return {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'features': features,
        'results': results
    }


# Train the XGBoost model
pipeline = train_classification_pipeline(
    'cup_dataset.csv', 
    target_column='fill',
    relevant_features = ['mfcc_5_slope', 'peak_frequency_mean', 'mfcc_6_slope', 'mfcc_7_max', 'zero_crossing_rate_slope','spectral_flatness_mean', 'mfcc_4_min', 'spectral_flatness_mean', 'mfcc_6_std', 'mfcc_5_median', 'mfcc_8_slope', 'rms_energy_std', 'mfcc_9_slope'],
    n_estimators=200,
    learning_rate=0.05
)

# # Single prediction
# new_sample = [5.1, 3.5, 1.4, 0.2, 0]
# prediction, confidence = predict_single(
#     pipeline['model'],
#     new_sample,
#     pipeline['scaler'],
#     pipeline['encoder']
# )
# print(f"Predicted class: {prediction} (confidence: {confidence:.2%})")

# # Batch predictions
# batch_samples = [[5.1, 3.5, 1.4, 0.2, 0], [6.2, 2.9, 4.3, 1.3, 1]]
# results = predict_batch(
#     pipeline['model'],
#     batch_samples,
#     pipeline['scaler'],
#     pipeline['encoder']
# )
# print(results)

# # Load model later
# model, scaler, encoder = load_model()

