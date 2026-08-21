from fastapi import APIRouter, HTTPException, status
import logging
from backend.app.schemas.predict import PatientVitals, RiskPredictionResponse
from backend.app.ml.pipeline import execute_pipeline
from backend.app.ml.fis.risk_bands import RiskFuzzyInference
from backend.app.ml.shap_explainer import generate_shap_values

router = APIRouter()
fis_engine = RiskFuzzyInference()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=RiskPredictionResponse)
def predict_risk(vitals: PatientVitals):
    """Generates risk prediction for a patient based on clinical parameters."""
    try:
        input_dict = vitals.model_dump()
        risk_prob = execute_pipeline(input_dict)
        fis_result = fis_engine.infer(risk_prob)
        
        # Calculate SHAP values for patient explanation, handle failures gracefully
        feature_contributions = []
        try:
            feature_contributions = generate_shap_values(input_dict)
        except Exception as e:
            logger.error(f"SHAP explanation calculation failed: {e}", exc_info=True)
            
        return RiskPredictionResponse(
            risk_probability=risk_prob,
            risk_band=fis_result["band"],
            confidence=fis_result["confidence"],
            feature_contributions=feature_contributions
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {str(e)}"
        )
