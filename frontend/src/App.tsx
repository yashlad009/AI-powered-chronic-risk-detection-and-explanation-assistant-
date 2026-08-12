import React, { useState } from 'react';
import { predictDiabetesRisk, PatientVitals, RiskPredictionResponse } from './api/predict';

type UiState = 'idle' | 'loading' | 'success' | 'error';

export default function App() {
  // Form states initialized with typical neutral clinical baselines
  const [pregnancies, setPregnancies] = useState<number>(0);
  const [glucose, setGlucose] = useState<number>(120);
  const [bloodPressure, setBloodPressure] = useState<number>(80);
  const [skinThickness, setSkinThickness] = useState<number>(20);
  const [insulin, setInsulin] = useState<number>(79);
  const [bmi, setBmi] = useState<number>(25.4);
  const [dpf, setDpf] = useState<number>(0.47);
  const [age, setAge] = useState<number>(33);

  // Status states
  const [status, setStatus] = useState<UiState>('idle');
  const [prediction, setPrediction] = useState<RiskPredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('loading');
    setErrorMsg('');

    const payload: PatientVitals = {
      Pregnancies: pregnancies,
      Glucose: glucose,
      BloodPressure: bloodPressure,
      SkinThickness: skinThickness,
      Insulin: insulin,
      BMI: bmi,
      DiabetesPedigreeFunction: dpf,
      Age: age
    };

    try {
      const data = await predictDiabetesRisk(payload);
      setPrediction(data);
      setStatus('success');
    } catch (err: any) {
      setErrorMsg(err.message || 'An unexpected error occurred.');
      setStatus('error');
    }
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="app-header">
        <div>
          <span className="app-eyebrow">Risk Assistant · hybrid ANN + FIS engine</span>
          <h1 className="app-title">Diabetes Prediction Requisition</h1>
        </div>
      </header>

      {/* Main Grid Layout */}
      <main className="dashboard-grid">
        {/* Left Panel: Input Requisition */}
        <section className="panel">
          <h2 className="panel-title">Patient Requisition</h2>
          
          <form onSubmit={handleSubmit} className="form-grid">
            <div className="form-group">
              <div className="label-row">
                <label htmlFor="pregnancies">Pregnancies</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="pregnancies"
                  type="number"
                  min="0"
                  step="1"
                  value={pregnancies}
                  onChange={(e) => setPregnancies(Math.max(0, parseInt(e.target.value) || 0))}
                  required
                />
                <span className="input-unit">count</span>
              </div>
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="glucose">Glucose</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="glucose"
                  type="number"
                  min="0"
                  step="1"
                  value={glucose}
                  onChange={(e) => setGlucose(Math.max(0, parseInt(e.target.value) || 0))}
                  required
                />
                <span className="input-unit">mg/dL</span>
              </div>
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="bloodPressure">Blood Pressure</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="bloodPressure"
                  type="number"
                  min="0"
                  step="1"
                  value={bloodPressure}
                  onChange={(e) => setBloodPressure(Math.max(0, parseInt(e.target.value) || 0))}
                  required
                />
                <span className="input-unit">mm Hg</span>
              </div>
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="skinThickness">Skin Thickness</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="skinThickness"
                  type="number"
                  min="0"
                  step="1"
                  value={skinThickness}
                  onChange={(e) => setSkinThickness(Math.max(0, parseInt(e.target.value) || 0))}
                  required
                />
                <span className="input-unit">mm</span>
              </div>
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="insulin">Insulin</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="insulin"
                  type="number"
                  min="0"
                  step="1"
                  value={insulin}
                  onChange={(e) => setInsulin(Math.max(0, parseInt(e.target.value) || 0))}
                  required
                />
                <span className="input-unit">mu U/ml</span>
              </div>
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="bmi">BMI</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="bmi"
                  type="number"
                  min="0.1"
                  step="0.1"
                  value={bmi}
                  onChange={(e) => setBmi(Math.max(0.1, parseFloat(e.target.value) || 0.1))}
                  required
                />
                <span className="input-unit">kg/m²</span>
              </div>
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="dpf">Pedigree Function</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="dpf"
                  type="number"
                  min="0"
                  step="0.001"
                  value={dpf}
                  onChange={(e) => setDpf(Math.max(0, parseFloat(e.target.value) || 0))}
                  required
                />
                <span className="input-unit">unitless</span>
              </div>
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="age">Age</label>
              </div>
              <div className="input-wrapper">
                <input
                  id="age"
                  type="number"
                  min="1"
                  step="1"
                  value={age}
                  onChange={(e) => setAge(Math.max(1, parseInt(e.target.value) || 1))}
                  required
                />
                <span className="input-unit">years</span>
              </div>
            </div>

            <button type="submit" className="btn-submit" disabled={status === 'loading'}>
              {status === 'loading' ? (
                <>
                  <span className="spinner"></span>
                  Processing metrics...
                </>
              ) : 'Submit Requisition'}
            </button>
          </form>
        </section>

        {/* Right Panel: Instrument Readout */}
        <section className="panel">
          <h2 className="panel-title">Instrument Readout</h2>
          
          <div className="readout-box">
            {status === 'idle' && (
              <div className="state-idle">
                <h3>Awaiting Requisition</h3>
                <p>Submit patient biometric data on the left panel to execute real-time model analysis.</p>
              </div>
            )}

            {status === 'loading' && (
              <div className="state-loading">
                <span className="spinner" style={{ borderColor: 'rgba(28, 35, 33, 0.2)', borderTopColor: 'var(--ink)', width: '24px', height: '24px', display: 'inline-block', marginBottom: '1rem' }}></span>
                <p>Evaluating ANN probabilities & FIS fuzzy parameters...</p>
              </div>
            )}

            {status === 'error' && (
              <div className="state-error">
                <h3 className="state-error-title">Analysis Failed</h3>
                <p>The prediction system returned the following exception/error:</p>
                <div className="state-error-detail">{errorMsg}</div>
              </div>
            )}

            {status === 'success' && prediction && (
              <div className="results-wrapper">
                {/* Score & Badge stamp */}
                <div className="results-header">
                  <div className="readout-numbers">
                    <div className="readout-num-group">
                      <span className="readout-num-label">Risk probability</span>
                      <span className="readout-num-val">{(prediction.risk_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="readout-num-group">
                      <span className="readout-num-label">FIS Confidence</span>
                      <span className="readout-num-val">{(prediction.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  
                  <div className={`badge-stamp ${prediction.risk_band.toLowerCase()}`}>
                    {prediction.risk_band}
                  </div>
                </div>

                {/* Strip-Chart Risk Gauge */}
                <div className="strip-chart-container">
                  <span className="readout-num-label">Risk Level Indicator</span>
                  <div className="strip-chart-track">
                    <div 
                      className="strip-chart-marker"
                      style={{ left: `${Math.min(100, Math.max(0, prediction.risk_probability * 100))}%` }}
                    />
                  </div>
                  <div className="strip-chart-ticks">
                    <div className="strip-chart-tick"><span>0.0</span></div>
                    <div className="strip-chart-tick"><span>0.25</span></div>
                    <div className="strip-chart-tick"><span>0.5</span></div>
                    <div className="strip-chart-tick"><span>0.75</span></div>
                    <div className="strip-chart-tick"><span>1.0</span></div>
                  </div>
                </div>

                {/* Clearly-marked Spot for Future Explanation Panel */}
                <div className="explanation-placeholder">
                  <div className="explanation-placeholder-header">Explanation & Feature Breakdown</div>
                  <div className="explanation-placeholder-content">
                    Model explanation module (SHAP/Feature Importance) is currently offline. Diagnostics breakdown will populate here when backend integration is established.
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
