from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

def save_model(
    model: xgb.XGBClassifier | RandomForestClassifier, 
    scaler: StandardScaler, 
    label_encoder: LabelEncoder, 
    selected_indices: list[int] | None = None,
    model_type: str = "model"
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
    model_path = f"{model_type}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'encoder': label_encoder,
            'selected_indices': selected_indices,
        }, f)
    print(f"Model saved to ")


def load_model(
    model_path: str,
) -> tuple[xgb.XGBClassifier | RandomForestClassifier, StandardScaler, LabelEncoder, list[int] | None]:
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
