import sqlite3
import joblib
import pandas as pd
import numpy as np
from utils.preprocessing import load_config, preprocess_data_inference

# --- 1. INITIALIZATION ---
# Load the specific inference configuration
config = load_config("./configs/inference_config.yaml")

# Select the model you want to run (must match a key in your YAML)
MODEL_NAME = "xgb_optimized" 

# Retrieve model-specific settings from YAML
model_cfg = config['models'][MODEL_NAME]
model_path = model_cfg['path']
threshold = model_cfg['threshold']
output_csv = model_cfg['output_csv']

# Load the model and the features list saved during training
model = joblib.load(model_path)
model_features = joblib.load("models/model_features.joblib")

def run_batch_inference(raw_df):
    """
    Complete inference pipeline: 
    Preprocessing -> Feature Alignment -> Probability Prediction -> Filtering
    """
    # 2. PREPROCESSING
    # Uses the shared logic to clean data and isolate customer IDs
    X_inference, ids = preprocess_data_inference(raw_df, config)
    
    if X_inference is None:
        print("No active customers found to process.")
        return None

    # 3. COLUMNS ALIGNMENT
    # Ensures the inference dataframe has the exact same columns as the training set
    for col in model_features:
        if col not in X_inference.columns:
            X_inference[col] = 0
            
    # Reorder columns to match the model's expected input order
    X_inference = X_inference[model_features]

    # 4. PREDICT PROBABILITIES
    # We take the probability of the positive class (Churn = 1)
    probabilities = model.predict_proba(X_inference)[:, 1]

    # 5. CREATE FINAL REPORT
    final_report = pd.DataFrame({
        'customerID': ids,
        'risk_score': probabilities
    })

    # 6. FILTER BY CUSTOM THRESHOLD
    # Sort by risk score (descending) to show the most critical cases first
    high_risk = final_report[final_report['risk_score'] >= threshold].sort_values(
        by='risk_score', ascending=False
    )
    
    return high_risk

if __name__ == "__main__":
    # Load database settings from YAML
    db_settings = config['database']
    
    # 7. CONNECT TO DATABASE
    print(f"Connecting to database: {db_settings['db_path']}...")
    conn = sqlite3.connect(db_settings['db_path'])
    
    try:
        # Load raw data
        table_sql = pd.read_sql(f"SELECT * FROM {db_settings['table_name']}", conn)
        
        # Run the inference engine
        high_risk_report = run_batch_inference(table_sql)

        if high_risk_report is not None:
            # Save results using the filename from YAML
            high_risk_report.to_csv(output_csv, index=False)
            print(f"✅ Success! Report saved as '{output_csv}'")
            print(f"Identified {len(high_risk_report)} customers at high risk (>= {threshold*100}%).")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        
    finally:
        conn.close()