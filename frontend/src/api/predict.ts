export interface PatientVitals {
  HighBP: number;               // binary flag (0 or 1), high blood pressure
  HighChol: number;             // binary flag (0 or 1), high cholesterol
  CholCheck: number;            // binary flag (0 or 1), cholesterol check in past 5 years
  BMI: number;                  // body mass index, > 0
  Smoker: number;               // binary flag (0 or 1), smoked at least 100 cigarettes
  Stroke: number;               // binary flag (0 or 1), ever had a stroke
  HeartDiseaseorAttack: number; // binary flag (0 or 1), heart disease or attack
  PhysActivity: number;         // binary flag (0 or 1), physical activity in past 30 days
  Fruits: number;               // binary flag (0 or 1), consume fruit daily
  Veggies: number;              // binary flag (0 or 1), consume vegetables daily
  HvyAlcoholConsump: number;    // binary flag (0 or 1), heavy alcohol consumption
  AnyHealthcare: number;        // binary flag (0 or 1), health care coverage
  NoDocbcCost: number;          // binary flag (0 or 1), could not see doctor due to cost
  GenHlth: number;              // general health scale, 1–5
  MentHlth: number;             // number of poor mental health days, 0–30
  PhysHlth: number;             // number of poor physical health days, 0–30
  DiffWalk: number;             // binary flag (0 or 1), difficulty walking
  Sex: number;                  // binary flag (0 = female, 1 = male)
  Age: number;                  // age bracket category, 1–13
  Education: number;            // education category level, 1–6
  Income: number;               // income category level, 1–8
}

export interface RiskPredictionResponse {
  risk_probability: number; // 0.0–1.0, raw ANN output
  risk_band: 'Low' | 'Medium' | 'High' | string; // from the Fuzzy Inference System
  confidence: number;       // 0.0–1.0, fuzzy membership confidence
}

/**
 * Sends patient vitals to the backend API to predict diabetes risk.
 * @param vitals Patient vitals matching the PatientVitals schema.
 * @returns Prediction response or throws an Error with details.
 */
export async function predictDiabetesRisk(vitals: PatientVitals): Promise<RiskPredictionResponse> {
  const response = await fetch('http://127.0.0.1:8000/api/v1/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(vitals),
  });

  if (!response.ok) {
    let errorMessage = `Server error: ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData && errorData.detail) {
        errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : JSON.stringify(errorData.detail);
      }
    } catch {
      // Fallback if parsing JSON fails
    }
    throw new Error(errorMessage);
  }

  return response.json();
}
