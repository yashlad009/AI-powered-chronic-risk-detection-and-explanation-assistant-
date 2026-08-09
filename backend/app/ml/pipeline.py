"""
Hybrid Prediction Pipeline.
Orchestrates the inputs through the ANN classifier, the Genetic-optimized params, and the Fuzzy Inference System.
"""

import os
import sys
import numpy as np
import joblib

# Add project root to sys.path to allow relative/absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.app.ml.ann.model import ChronicRiskANN

# Define paths relative to this file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
model_path = os.path.join(base_dir, "ml-training", "saved_models", "ann_model.keras")
scaler_path = os.path.join(base_dir, "ml-training", "saved_models", "scaler.joblib")
imputation_path = os.path.join(base_dir, "ml-training", "saved_models", "imputation_values.joblib")

_model = None
_scaler = None
_imputation_values = None

def _load_resources():
    """Helper function to load model, scaler, and imputation values lazily."""
    global _model, _scaler, _imputation_values
    if _model is None or _scaler is None or _imputation_values is None:
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(imputation_path):
            _model = ChronicRiskANN(input_dim=8, dropout_rate=0.2)
            _model.load(model_path)
            _scaler = joblib.load(scaler_path)
            _imputation_values = joblib.load(imputation_path)
        else:
            raise FileNotFoundError(
                f"Required resources not found. Ensure training is complete. "
                f"Missing one or more of:\n- '{model_path}'\n- '{scaler_path}'\n- '{imputation_path}'"
            )

def predict(patient_dict):
    """
    Predicts the raw chronic disease risk probability for a patient.
    Uses saved train-only imputation values for missing (0) features.
    
    Args:
        patient_dict (dict): Dictionary containing patient metrics with keys:
            - Pregnancies
            - Glucose
            - BloodPressure
            - SkinThickness
            - Insulin
            - BMI
            - DiabetesPedigreeFunction
            - Age

    Returns:
        float: Raw probability of chronic disease risk.
    """
    _load_resources()

    feature_order = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    # Make a copy of the input dictionary to avoid side effects
    processed_patient = patient_dict.copy()

    # Apply saved train-only imputation values for columns where 0 represents a missing value
    target_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in target_cols:
        val = processed_patient.get(col, 0)
        # If value is 0, None, or NaN, replace it with the train-set imputation value
        if val == 0 or val is None or (isinstance(val, float) and np.isnan(val)):
            processed_patient[col] = _imputation_values[col]

    # Extract features in the correct order
    features = np.array([[processed_patient[feat] for feat in feature_order]], dtype=float)

    # Scale the features using the fitted scaler
    features_scaled = _scaler.transform(features)

    # Get prediction probability from the ANN model
    prob = _model.predict(features_scaled)

    return float(prob[0][0])

def execute_pipeline(input_data):
    """Executes the end-to-end hybrid classification prediction pipeline."""
    return predict(input_data)
