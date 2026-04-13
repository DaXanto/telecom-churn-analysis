import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
from utils.preprocessing import load_config,load_data,preprocess_data_training

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

    # Load database settings from YAML
    db_settings = config['database']
    
    print(f"Connecting to database: {db_settings['db_path']}...")
    conn = sqlite3.connect(db_settings['db_path'])

    table_sql = pd.read_sql(f"SELECT * FROM {db_settings['table_name']}", conn)

    # 2. Preprocessing
    df_final = preprocess_data_training(table_sql, config)
    
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

    # Save features
    joblib.dump(X_train.columns.tolist(), "models/model_features.joblib")

    # 5. Training loop for models defined in config
    for model_id, m_cfg in config['models'].items():
        print(f"\n" + "="*40)
        print(f" TRAINING: {model_id.upper()}")
        print("="*40)

        # Instance and Fit
        model = get_model_instance(m_cfg['type'], m_cfg['params'])
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model,f"models/{model_id}.joblib")
        print(f"Modèle sauvegardé : models/{model_id}.joblib")

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