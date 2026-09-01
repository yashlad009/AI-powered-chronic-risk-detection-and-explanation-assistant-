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
model_path = os.path.join(base_dir, "ml-training", "saved_models", "ann_model_brfss.keras")
scaler_path = os.path.join(base_dir, "ml-training", "saved_models", "scaler_brfss.joblib")

_model = None
_scaler = None

def _load_resources():
    """Helper function to load model and scaler lazily."""
    global _model, _scaler
    if _model is None or _scaler is None:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            _model = ChronicRiskANN(input_dim=21, dropout_rate=0.2)
            _model.load(model_path)
            _scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(
                f"Required resources not found. Ensure training is complete. "
                f"Missing one or more of:\n- '{model_path}'\n- '{scaler_path}'"
            )

def predict(patient_dict):
    """
    Predicts the raw chronic disease risk probability for a patient.
    
    Args:
        patient_dict (dict): Dictionary containing patient metrics with keys:
            - HighBP
            - HighChol
            - CholCheck
            - BMI
            - Smoker
            - Stroke
            - HeartDiseaseorAttack
            - PhysActivity
            - Fruits
            - Veggies
            - HvyAlcoholConsump
            - AnyHealthcare
            - NoDocbcCost
            - GenHlth
            - MentHlth
            - PhysHlth
            - DiffWalk
            - Sex
            - Age
            - Education
            - Income

    Returns:
        float: Raw probability of chronic disease risk.
    """
    _load_resources()

    feature_order = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income"
    ]

    # Make a copy of the input dictionary to avoid side effects
    processed_patient = patient_dict.copy()

    # Extract features in the correct order
    features = np.array([[processed_patient.get(feat, 0.0) for feat in feature_order]], dtype=float)

    # Scale the features using the fitted scaler
    features_scaled = _scaler.transform(features)

    # Get prediction probability from the ANN model
    prob = _model.predict(features_scaled)

    return float(prob[0][0])

def execute_pipeline(input_data):
    """Executes the end-to-end hybrid classification prediction pipeline."""
    return predict(input_data)
