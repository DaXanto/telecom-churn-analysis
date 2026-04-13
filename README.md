# Customer Churn Prediction Pipeline (Professional ML Template)

This repository contains a clean, production-ready Machine Learning pipeline designed to predict customer churn. The project is fully modular, using a YAML configuration system to handle preprocessing, model selection, and evaluation.

---

## Table of Contents
1. [Installation](#-installation)
2. [Project Architecture](#-project-architecture)
3. [Configuration Guide (YAML)](#-configuration-guide-yaml)
4. [Pipeline Logic](#-pipeline-logic)
5. [Evaluation Metrics](#-evaluation-metrics)
6. [How to run](#-how-to-run)

---

## Installation

### 1. Clone & Setup
```bash
# Clone the project
git clone <https://github.com/DaXanto/telecom-churn-analysis.git>
cd <project-folder>

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
---
### 2. Project Architecture

The project follows a modular structure to separate logic from configuration:

```text
.
├── data/
│   └── churn_telecom.csv    # Raw dataset (input source)
├── config.yaml              # Centralized configuration (The "Brain")
├── main.py                  # Main execution script (The "Engine")
├── requirements.txt         # Project dependencies (The "Environment")
└── README.md                # Project documentation
```
---

## 3. Configuration Guide (YAML)

The `config.yaml` file is the central control panel of the project. It allows you to modify the behavior of the pipeline without touching the Python code.

### Data Section
Defines where and how to load your dataset.
* **`path`**: Relative path to your CSV file.
* **`sep`**: The character used to separate columns (e.g., `,` or `;`).
* **`encoding`**: File encoding (standard is `utf-8`).

### Preprocessing Section
Controls how data is cleaned and transformed.
* **`binary_mapping`**: Specifically maps text categories (like "Yes"/"No") to numeric `1` and `0`.
* **`features_to_create`**: 
    * **`name`**: The name of the new column to be created.
    * **`cols`**: The list of existing columns to aggregate (counts the occurrences of "Yes").
* **`drop_na`**: If `true`, removes all rows with missing values before training.

### Models Section
This is where you define your experiments. You can add as many model blocks as you like:
* **`type`**: The class name of the model (`RandomForestClassifier` or `XGBClassifier`).
* **`threshold`**: The classification threshold. 
    * *Standard is 0.5.* * *Increase it (e.g., 0.7) to be more "sure" before predicting a Churn, reducing False Positives.*
* **`params`**: Any valid hyperparameter accepted by `scikit-learn` or `XGBoost`.

### SMOTE Section
* **`enabled`**: Set to `true` to use Synthetic Minority Over-sampling Technique, which helps the model learn better if you have very few "Churn" examples compared to loyal customers.

---

## 4. Pipeline Logic

The `main.py` script follows a structured execution flow to transform raw data into evaluated models. Here is the step-by-step logic:

### Step 1: Data Loading & Initialization
The pipeline starts by reading the `config.yaml`. It uses the `data` parameters to load the CSV file into a Pandas DataFrame. If the file is missing or the encoding is wrong, the script provides a clear error message.

### Step 2: Automated Preprocessing
Instead of manual cleaning, the pipeline applies a dynamic transformation:
1.  **Feature Engineering**: It creates new features (like `Service_Count`) by aggregating columns specified in the config.
2.  **Binary Encoding**: It scans for columns with only 2 unique values and maps them using your `binary_mapping` (e.g., `Yes` → `1`).
3.  **One-Hot Encoding**: All other categorical variables are converted into dummy variables (`pd.get_dummies`), making them readable for the algorithms.
4.  **Type Consistency**: All boolean outputs are forced to `int64` to prevent compatibility issues with XGBoost or Scikit-Learn.

### Step 3: Resampling (SMOTE)
If `smote: enabled` is set to `true`, the pipeline uses the **Synthetic Minority Over-sampling Technique**. This creates synthetic examples of the "Churn" class to balance the training set, preventing the model from simply "guessing" that everyone stays just because they are the majority.

### Step 4: The Model Factory
The script iterates through the `models` dictionary in your YAML:
* It **instantiates** the model dynamically based on the `type` string.
* It **injects** the hyperparameters defined in `params`.
* It **fits** the model on the (resampled) training data.

### Step 5: Custom Probability Thresholding
Standard models predict at a `0.5` threshold. Our pipeline uses `predict_proba()` to apply the custom `threshold` defined in the config. 
* **Why?** In Churn prediction, it is often better to have a high threshold (e.g., 0.7) to target only customers with a very high probability of leaving, ensuring marketing budgets are spent wisely.

---

## 5. Evaluation Metrics

To ensure the models are reliable and actionable, the pipeline generates a comprehensive set of metrics after each training session.

### Standard Classification Metrics
For every model defined in your configuration, the console will output:

* **Accuracy Score**: The overall percentage of correct predictions. While useful, it can be misleading if your classes are highly imbalanced (e.g., if only 10% of customers churn).
* **Precision**: Out of all customers the model predicted as "Churn", how many actually left? High precision means fewer false alarms.
* **Recall (Sensitivity)**: Out of all customers who actually left, how many did the model correctly identify? High recall is crucial if you want to catch as many potential churners as possible.
* **F1-Score**: The harmonic mean of Precision and Recall, providing a single score to balance both metrics.

### Confusion Matrix
The pipeline prints a confusion matrix for each model, allowing you to see exactly:
- **True Positives (TP)**: Correctly predicted churners.
- **True Negatives (TN)**: Correctly predicted loyal customers.
- **False Positives (FP)**: Loyal customers wrongly predicted as churners (Type I error).
- **False Negatives (FN)**: Churners the model missed (Type II error).

### Feature Importance Visualization
For tree-based models (like Random Forest), the script automatically generates a bar chart using `Seaborn`.



* **What it tells you**: It identifies which variables (e.g., `tenure`, `MonthlyCharges`, `Service_Count`) have the most significant impact on the model's decision-making process.
* **Business Value**: This helps stakeholders understand *why* customers are leaving, allowing for targeted business strategy improvements.

---

## 6. How to Run

Follow these steps in order to ensure the data is ready and the models are trained correctly.

### Step 1: Preliminary Data Processing
Before running the main pipeline, you must execute the data cleaning notebook. This script handles initial raw data formatting and exports the cleaned CSV used by the engine.

* Open and run all cells in: `data_processing.ipynb`
* This will generate the processed file in the `data/` folder.

### Step 2: train Pipeline Execution
Once the cleaned data is ready, run the following command in your terminal:

```bash
python main.py
```

### Step 2: Main Pipeline Execution

```bash
python inference.py
```