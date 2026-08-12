export interface PatientVitals {
  Pregnancies: number;              // integer, >= 0
  Glucose: number;                  // mg/dL, >= 0
  BloodPressure: number;            // diastolic, mm Hg, >= 0
  SkinThickness: number;            // mm, >= 0
  Insulin: number;                  // mu U/ml, >= 0
  BMI: number;                      // kg/m^2, > 0
  DiabetesPedigreeFunction: number; // >= 0
  Age: number;                      // integer, > 0
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
