from pydantic import BaseModel, Field

class PatientVitals(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="High blood pressure (0 = no, 1 = yes)")
    HighChol: int = Field(..., ge=0, le=1, description="High cholesterol (0 = no, 1 = yes)")
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol check in past 5 years (0 = no, 1 = yes)")
    BMI: float = Field(..., gt=0, description="Body Mass Index")
    Smoker: int = Field(..., ge=0, le=1, description="Have smoked at least 100 cigarettes in entire life (0 = no, 1 = yes)")
    Stroke: int = Field(..., ge=0, le=1, description="Ever told you had a stroke (0 = no, 1 = yes)")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Coronary heart disease or myocardial infarction (0 = no, 1 = yes)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity in past 30 days (0 = no, 1 = yes)")
    Fruits: int = Field(..., ge=0, le=1, description="Consume fruit 1 or more times per day (0 = no, 1 = yes)")
    Veggies: int = Field(..., ge=0, le=1, description="Consume vegetables 1 or more times per day (0 = no, 1 = yes)")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Heavy alcohol consumption (0 = no, 1 = yes)")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Have any health care coverage (0 = no, 1 = yes)")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Could not see doctor because of cost in past 12 months (0 = no, 1 = yes)")
    GenHlth: int = Field(..., ge=1, le=5, description="Self-rated general health (1 = excellent, 5 = poor)")
    MentHlth: int = Field(..., ge=0, le=30, description="Number of days of poor mental health in past 30 days")
    PhysHlth: int = Field(..., ge=0, le=30, description="Number of days of poor physical health in past 30 days")
    DiffWalk: int = Field(..., ge=0, le=1, description="Serious difficulty walking or climbing stairs (0 = no, 1 = yes)")
    Sex: int = Field(..., ge=0, le=1, description="Sex (0 = female, 1 = male)")
    Age: int = Field(..., ge=1, le=13, description="13-level age category (1 = 18-24, 13 = 80+)")
    Education: int = Field(..., ge=1, le=6, description="Education level (1 = elementary, 6 = graduate)")
    Income: int = Field(..., ge=1, le=8, description="Income scale (1 = <$10k, 8 = >=$75k)")

class RiskPredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Raw risk probability of chronic disease")
    risk_band: str = Field(..., description="Qualitative risk band based on FIS (Low, Medium, High)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fuzzy membership confidence score")
