import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.feature_selection import f_classif

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
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Preprocess data: handle missing values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw dataset
    target_column : str
        Name of the target column (classification labels)
        
    Returns:
    --------
    tuple
        (X, y, feature_names)
    """
    # Handle missing values
    #data = data.dropna()

    # Resolve column names case-insensitively so 'Fill'/'fill' and 'Cup'/'cup'
    # both work regardless of how the CSV is capitalized.
    lookup = {c.lower(): c for c in data.columns}

    def _col(name: str) -> str:
        key = name.lower()
        if key not in lookup:
            raise KeyError(
                f"column '{name}' not found in data; available columns include: "
                f"{list(data.columns)[:5]}..."
            )
        return lookup[key]

    target_col = _col(target_column)
    cup_col = _col('cup')
    duration_col = _col('duration_s')

    # Separate features and target
    X = data.drop(columns=[target_col, cup_col])
    X.drop(columns=[duration_col], inplace=True)
    X.drop(X.columns[X.nunique() < 2], axis=1, inplace=True)

    y = data[target_col]

    cup_labels = data[cup_col]
    return X, y, cup_labels, X.columns.tolist()
    
def encode_target_to_classes(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder

def split_data(X : pd.DataFrame ,y_encoded, test_size: float = 0.2, random_state: int = 42):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded, shuffle=True
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {np.unique(y_encoded)}")
    
    return X_train, X_test, y_train, y_test

def split_data_by_cup(X : pd.DataFrame ,y_encoded, cup_labels, cup_number):
    mask = cup_labels == cup_number
    X_test = X[mask]
    X_train = X[~mask]
    y_test = y_encoded[mask]
    y_train = y_encoded[~mask]

    return X_train, X_test, y_train, y_test

def split_data_with_indices(X: pd.DataFrame, y_encoded, test_size: float = 0.2, random_state: int = 42):
    """
    Same as split_data, but also returns the original row indices for the
    train and test sets so downstream code can map test rows back to the CSV.

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, train_idx, test_idx)
        where train_idx / test_idx are positional row indices into X.
    """
    indices = np.arange(len(X))
    (X_train, X_test, y_train, y_test, train_idx, test_idx) = train_test_split(
        X, y_encoded, indices,
        test_size=test_size, random_state=random_state,
        stratify=y_encoded, shuffle=True,
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {np.unique(y_encoded)}")
    return X_train, X_test, y_train, y_test, np.asarray(train_idx), np.asarray(test_idx)


def split_data_by_cup_with_indices(X: pd.DataFrame, y_encoded, cup_labels, cup_number):
    """
    Same as split_data_by_cup (leave-one-cup-out), but also returns the
    original positional row indices for the train and test sets.
    """
    cup_labels = np.asarray(cup_labels)
    indices = np.arange(len(X))
    mask = cup_labels == cup_number
    X_test = X[mask]
    X_train = X[~mask]
    y_test = y_encoded[mask]
    y_train = y_encoded[~mask]
    test_idx = indices[mask]
    train_idx = indices[~mask]
    return X_train, X_test, y_train, y_test, train_idx, test_idx


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

def anova_f_scores(X: pd.DataFrame, y_encoded) -> pd.DataFrame:
    """
    Compute ANOVA F-value and p-value for each numeric feature in X vs categorical target y.
    Returns a DataFrame sorted by descending F-value with columns: feature, F, p.

    - X: feature dataframe (only numeric columns are tested).
    - y: target vector (pd.Series or 1D np.ndarray). Non-numeric targets are label-encoded.
    - drop_na: if True, rows with NA in the feature or target are dropped for that feature.
    """
    if isinstance(y_encoded, np.ndarray):
        y_encoded = pd.Series(y_encoded)
    if len(y_encoded) != len(X):
        raise ValueError("X and y must have the same number of rows")

    # Select numeric feature columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("no numeric feature columns found to test")

    features, f_vals, p_vals = [], [], []
    for col in numeric_cols:
        xi_df = pd.concat([X[col], y_encoded], axis=1)
        xi_df.columns = [col, "target"]
        if xi_df.empty:
            continue
        xi_X = xi_df[col].values.reshape(-1, 1)
        xi_y = xi_df["target"]
        # encode target per-feature if still non-numeric (safeguard)
        if xi_y.dtype.kind not in "iuf":
            xi_y = LabelEncoder().fit_transform(xi_y.astype(str))
        f, p = f_classif(xi_X, xi_y)
        features.append(col)
        f_vals.append(float(f[0]))
        p_vals.append(float(p[0]))

    result = pd.DataFrame({"feature": features, "F": f_vals, "p": p_vals})
    result = result.sort_values("F", ascending=False).reset_index(drop=True)
    return result