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

class RiskPredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Raw risk probability of chronic disease")
    risk_band: str = Field(..., description="Qualitative risk band based on FIS (Low, Medium, High)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fuzzy membership confidence score")
