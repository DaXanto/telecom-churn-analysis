import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# --- CONFIGURATION AND LOADING ---

def load_config(path="./config/config.yaml"):
    """Load the YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file '{path}' not found.")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_data(config):
    """Load the CSV dataset based on the config parameters."""
    cfg = config['data']
    print(f"Loading data from: {cfg['path']}")
    return pd.read_csv(cfg['path'], sep=cfg['sep'], encoding=cfg['encoding'])

# --- PREPROCESSING ---

def preprocess_data(df, config):
    """Clean, Feature Engineering, and Encoding."""
    cfg_pre = config['preprocessing']
    target = cfg_pre['target_column']
    mapping = cfg_pre['binary_mapping']
    
    df_processed = df.copy()

    # 1. Feature Engineering (Service Count)
    for feat in cfg_pre.get('features_to_create', []):
        df_processed[feat['name']] = df_processed[feat['cols']].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )
        print(f"Feature created: {feat['name']}")

    # 2. Encoding: Binary (Map) vs One-Hot (Dummies)
    cols_to_encode = []
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col == target:
            continue
        
        # If the column has only 2 unique values, apply binary mapping
        if df_processed[col].nunique() == 2:
            df_processed[col] = df_processed[col].map(mapping)
        else:
            cols_to_encode.append(col)

    # 3. Apply One-Hot Encoding for multi-categorical variables
    df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=True)
    
    # 4. Target Encoding
    df_processed[target] = df_processed[target].map(mapping)
    
    # 5. Convert booleans (from dummies) to int64
    bool_cols = df_processed.select_dtypes('bool').columns
    df_processed[bool_cols] = df_processed[bool_cols].astype('int64')

    # 6. Final Cleaning
    if cfg_pre.get('drop_na', True):
        df_processed = df_processed.dropna()
        
    return df_processed.reset_index(drop=True)

# --- TRAINING AND EVALUATION ---

def get_model_instance(model_type, params):
    """Factory to instantiate the correct model."""
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    elif model_type == "XGBClassifier":
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Model type '{model_type}' not recognized.")

def run_pipeline():
    # 1. Initialization
    config = load_config()
    df_raw = load_data(config)
    
    # 2. Preprocessing
    df_final = preprocess_data(df_raw, config)
    
    # 3. Train/Test Split
    target = config['global']['target']
    X = df_final.drop(columns=[target])
    y = df_final[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['global']['test_size'], 
        random_state=config['global']['random_state']
    )

    # 4. Class Balancing (SMOTE)
    if config['smote']['enabled']:
        print("Applying SMOTE...")
        sm = SMOTE(random_state=config['smote']['random_state'])
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # 5. Training loop for models defined in config
    for model_id, m_cfg in config['models'].items():
        print(f"\n" + "="*40)
        print(f" TRAINING: {model_id.upper()}")
        print("="*40)

        # Instance and Fit
        model = get_model_instance(m_cfg['type'], m_cfg['params'])
        model.fit(X_train, y_train)

        # Prediction with custom Threshold
        threshold = m_cfg.get('threshold', 0.5)
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs > threshold).astype(int)

        # Display results
        print(f"Decision Threshold: {threshold}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
        
        # Optional: Feature Importance for Random Forest
        if m_cfg['type'] == "RandomForestClassifier":
            plot_importance(model, X.columns, model_id)

def plot_importance(model, feature_names, model_name):
    """Generate a feature importance bar plot."""
    importance_df = pd.DataFrame({
        'Feature': feature_names, 
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="magma")
    plt.title(f"Top 15 Features - {model_name}")
    plt.tight_layout()
    plt.show()

# --- ENTRY POINT ---

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"FATAL ERROR: {e}")