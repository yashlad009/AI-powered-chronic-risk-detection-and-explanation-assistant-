from pydantic import BaseModel, Field

class PatientVitals(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, description="Glucose level (mg/dL)")
    BloodPressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., gt=0, description="Body Mass Index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function score")
    Age: int = Field(..., gt=0, description="Age in years")

class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Original vitals value for this feature")
    shap_value: float = Field(..., description="SHAP value indicating contribution to risk probability")

class RiskPredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Raw risk probability of chronic disease")
    risk_band: str = Field(..., description="Qualitative risk band based on FIS (Low, Medium, High)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fuzzy membership confidence score")
    feature_contributions: list[FeatureContribution] = Field(
        default=[],
        description="Feature contribution explanations calculated via SHAP, sorted by descending absolute impact"
    )

