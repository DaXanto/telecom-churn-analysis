import os
import yaml
import pandas as pd
import numpy as np

def load_config(path="./configs/training_config.yaml"):
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

def preprocess_data_training(df, config):
    """Clean, Feature Engineering, and Encoding."""
    cfg_pre = config['preprocessing']
    target = cfg_pre['target_column']
    mapping = cfg_pre['binary_mapping']
    
    df_processed = df.copy()

    df_processed = df_processed.replace(r'^\s*$', np.nan, regex=True)

    df_processed = df_processed.drop(columns=["customerID"])
    df_processed = df_processed.drop_duplicates()
    df_processed = df_processed.dropna()

    df_processed.reset_index(drop=True, inplace=True)

    colonnes_to_keep = cfg_pre["features_to_use"]

    for col in df_processed.columns:
        if col not in colonnes_to_keep:
            df_processed.drop(columns=col, inplace=True)
    
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

def preprocess_data_inference(df, config):
    """
    Prepare data for prediction: 
    - Filters active customers only
    - Isolates IDs
    - Cleans and encodes features
    """
    cfg_pre = config['preprocessing']
    target = cfg_pre['target_column']
    mapping = cfg_pre['binary_mapping']
    
    # 1. Filter exclusively customers who have not churned yet
    if target in df.columns:
        # We only want to predict for current customers (Churn == 'No')
        df = df[df[target].astype(str).str.lower() == 'no'].copy()
    else:
        # If the target column is missing (e.g., new data), assume all are active
        df = df.copy()

    if df.empty:
        return None, None

    # 2. Extract and store Customer IDs for final reporting
    customer_ids = df['customerID'].reset_index(drop=True)
    
    # 3. Initial cleaning: handle empty strings and numeric conversion
    df_processed = df.replace(r'^\s*$', np.nan, regex=True)
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

    # 4. Feature selection based on the YAML configuration
    columns_to_keep = cfg_pre["features_to_use"]
    df_processed = df_processed[columns_to_keep]

    # 5. Feature Engineering (e.g., counting active services)
    for feat in cfg_pre.get('features_to_create', []):
        if all(c in df_processed.columns for c in feat['cols']):
            df_processed[feat['name']] = df_processed[feat['cols']].apply(
                lambda x: (x == 'Yes').sum(), axis=1
            )

    # 6. Binary Encoding (Map columns listed in config)
    binary_cols = cfg_pre.get('binary_columns', [])
    cols_to_one_hot = []
    
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col in binary_cols:
            df_processed[col] = df_processed[col].map(mapping)
        else:
            # Multi-categorical columns will be processed via One-Hot Encoding
            cols_to_one_hot.append(col)

    # 7. Apply One-Hot Encoding for remaining categorical variables
    df_processed = pd.get_dummies(df_processed, columns=cols_to_one_hot, drop_first=True)
    
    # 8. Type conversion: Ensure boolean columns are cast to int64 for model compatibility
    bool_cols = df_processed.select_dtypes('bool').columns
    df_processed[bool_cols] = df_processed[bool_cols].astype('int64')

    # Note: We do NOT process the target here as it is either 'No' or absent during inference
    
    return df_processed.reset_index(drop=True), customer_ids