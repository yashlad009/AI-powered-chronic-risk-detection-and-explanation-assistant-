import os
import numpy as np
import pandas as pd
import shap
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Module-level globals for caching the background sample and the explainer
_explainer = None
_background_data = None

def _init_explainer():
    """Initializes the SHAP KernelExplainer lazily to optimize startup latency."""
    global _explainer, _background_data
    if _explainer is not None:
        return

    try:
        from backend.app.ml import pipeline
        # Load resources from pipeline.py if not already loaded
        pipeline._load_resources()
        
        # Verify pipeline resources are active
        if pipeline._model is None or pipeline._scaler is None or pipeline._imputation_values is None:
            raise ValueError("Pipeline resources (model, scaler, imputation_values) are not loaded.")

        # Load training dataset for background reference
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        csv_path = os.path.join(base_dir, "ml-training", "datasets", "diabetes.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training dataset not found at '{csv_path}'.")

        df = pd.read_csv(csv_path)
        if "Outcome" in df.columns:
            df = df.drop(columns=["Outcome"])

        # Take a small background sample (e.g. 100 rows) for computation speed
        bg_sample = df.head(100).copy()

        # Preprocess features using the same order and imputation as pipeline.py
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

        target_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in target_cols:
            bg_sample[col] = bg_sample[col].apply(
                lambda val: pipeline._imputation_values[col] if (val == 0 or val is None or (isinstance(val, float) and np.isnan(val))) else val
            )

        # Reorder columns to ensure consistency
        bg_sample = bg_sample[feature_order]

        # Scale features using the fitted scaler
        bg_scaled = pipeline._scaler.transform(bg_sample.values)
        _background_data = bg_scaled

        # Define wrapper for prediction output flattening
        def model_predict(x):
            # Ensure tensorflow predictions are quiet by using the underlying model
            preds = pipeline._model.model.predict(x, verbose=0)
            return preds.flatten()

        # Build KernelExplainer
        _explainer = shap.KernelExplainer(model_predict, bg_scaled)
        logger.info("SHAP KernelExplainer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SHAP explainer: {e}", exc_info=True)
        raise

def generate_shap_values(input_dict):
    """
    Calculates SHAP values for a single patient's 8 features against the loaded ANN model.
    
    Args:
        input_dict (dict): Dictionary containing patient vitals.
        
    Returns:
        list[dict]: List of {"feature": str, "value": float, "shap_value": float}
                    sorted by |shap_value| descending.
    """
    # Ensure explainer is initialized
    _init_explainer()
    
    from backend.app.ml import pipeline

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

    # Process and impute input vitals identical to pipeline.py
    processed_patient = input_dict.copy()
    target_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in target_cols:
        val = processed_patient.get(col, 0)
        if val == 0 or val is None or (isinstance(val, float) and np.isnan(val)):
            processed_patient[col] = pipeline._imputation_values[col]

    # Shape features and scale
    features = np.array([[processed_patient[feat] for feat in feature_order]], dtype=float)
    features_scaled = pipeline._scaler.transform(features)

    # Compute SHAP values with small nsamples for latency
    shap_vals = _explainer.shap_values(features_scaled, nsamples=100, silent=True)

    # Resolve return format if list returned
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    shap_vals = np.squeeze(shap_vals)

    # Pair features with values and their SHAP contributions
    results = []
    for i, feat in enumerate(feature_order):
        results.append({
            "feature": feat,
            "value": float(input_dict.get(feat, processed_patient[feat])),
            "shap_value": float(shap_vals[i])
        })

    # Sort descending by absolute SHAP value impact
    results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    return results

