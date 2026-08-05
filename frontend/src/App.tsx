import React, { useState } from 'react';

interface RiskFactor {
  name: string;
  weight: number; // 0 to 100
  impact: 'High' | 'Medium' | 'Low';
}

export default function App() {
  const [patientId, setPatientId] = useState('');
  const [age, setAge] = useState(45);
  const [bmi, setBmi] = useState(27.4);
  const [systolic, setSystolic] = useState(130);
  const [diastolic, setDiastolic] = useState(85);
  const [cholesterol, setCholesterol] = useState(210);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prediction, setPrediction] = useState<{
    riskScore: number;
    category: string;
    factors: RiskFactor[];
    recommendations: string[];
  } | null>(null);

  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API response
    setTimeout(() => {
      // Calculate a pseudo risk score based on simple metrics
      const scoreWeight = (age * 0.4) + ((bmi - 18.5) * 1.5) + ((systolic - 120) * 0.5) + ((cholesterol - 150) * 0.2);
      const score = Math.min(Math.max(Math.round(scoreWeight), 5), 98);
      
      let category = 'Low Risk';
      if (score > 60) category = 'High Risk';
      else if (score > 35) category = 'Moderate Risk';

      const factors: RiskFactor[] = [
        { name: 'Systolic Blood Pressure', weight: Math.min(Math.round((systolic / 180) * 100), 100), impact: systolic > 140 ? 'High' : systolic > 130 ? 'Medium' : 'Low' },
        { name: 'Body Mass Index (BMI)', weight: Math.min(Math.round((bmi / 40) * 100), 100), impact: bmi > 30 ? 'High' : bmi > 25 ? 'Medium' : 'Low' },
        { name: 'Total Cholesterol', weight: Math.min(Math.round((cholesterol / 300) * 100), 100), impact: cholesterol > 240 ? 'High' : cholesterol > 200 ? 'Medium' : 'Low' },
        { name: 'Age Factor', weight: Math.min(Math.round((age / 100) * 100), 100), impact: age > 65 ? 'High' : age > 45 ? 'Medium' : 'Low' }
      ].sort((a, b) => b.weight - a.weight);

      const recommendations = [];
      if (bmi > 25) recommendations.push('Adopt a heart-healthy diet low in saturated fats and refined sugars to help optimize body mass.');
      if (systolic > 130 || diastolic > 80) recommendations.push('Incorporate 150 minutes of moderate cardiovascular activity weekly to lower blood pressure naturally.');
      if (cholesterol > 200) recommendations.push('Consult a healthcare professional about managing lipid profiles with omega-3 rich nutrition or targeted therapies.');
      if (recommendations.length === 0) recommendations.push('Maintain your excellent healthy lifestyle and schedule regular annual biometric screenings.');

      setPrediction({
        riskScore: score,
        category,
        factors,
        recommendations
      });
      setIsSubmitting(false);
    }, 1200);
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <span className="logo-icon">🏥</span>
          <div>
            <h1>Cardiologix AI</h1>
            <p className="subtitle">Chronic Risk Detection & Explanation Engine</p>
          </div>
        </div>
        <div className="api-badge">
          <span className="status-dot"></span>
          <span>Core Agent Sync Active</span>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Left Form Panel */}
        <section className="card form-card">
          <h2 className="card-title">Biometric Entry</h2>
          <p className="card-subtitle">Input real-time patient statistics to compute risk metrics</p>
          
          <form onSubmit={handlePredict}>
            <div className="form-group">
              <label>Patient Identifier</label>
              <input 
                type="text" 
                placeholder="e.g. PAT-9082" 
                value={patientId}
                onChange={e => setPatientId(e.target.value)}
                required 
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Age (years)</label>
                <input 
                  type="number" 
                  min="1" 
                  max="120" 
                  value={age} 
                  onChange={e => setAge(Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label>BMI (kg/m²)</label>
                <input 
                  type="number" 
                  step="0.1" 
                  min="10" 
                  max="60" 
                  value={bmi} 
                  onChange={e => setBmi(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Systolic BP (mmHg)</label>
                <input 
                  type="number" 
                  min="70" 
                  max="250" 
                  value={systolic} 
                  onChange={e => setSystolic(Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label>Diastolic BP (mmHg)</label>
                <input 
                  type="number" 
                  min="40" 
                  max="150" 
                  value={diastolic} 
                  onChange={e => setDiastolic(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="form-group">
              <label>Total Cholesterol (mg/dL)</label>
              <input 
                type="number" 
                min="100" 
                max="500" 
                value={cholesterol} 
                onChange={e => setCholesterol(Number(e.target.value))}
              />
            </div>

            <button type="submit" className="btn-primary" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <span className="spinner"></span>
                  Processing Models...
                </>
              ) : 'Run Analysis Engine'}
            </button>
          </form>
        </section>

        {/* Right Output Panel */}
        <section className="card output-card">
          {!prediction && !isSubmitting && (
            <div className="empty-state">
              <div className="empty-icon">📊</div>
              <h3>Analysis Awaiting Biometrics</h3>
              <p>Fill out the patient metrics on the left panel to execute chronic disease prediction algorithms and get detailed explanations.</p>
            </div>
          )}

          {isSubmitting && (
            <div className="loading-state">
              <div className="loading-logo">⚡</div>
              <h3>Calculating Risk Scores</h3>
              <p>Analyzing biomarkers against model cohorts...</p>
            </div>
          )}

          {prediction && !isSubmitting && (
            <div className="prediction-results">
              <h2 className="card-title">Diagnostics Report</h2>
              
              {/* Risk Gauge Header */}
              <div className="risk-score-display">
                <div className="score-ring">
                  <div className="score-value">{prediction.riskScore}%</div>
                  <div className="score-label">Risk Index</div>
                </div>
                <div className="status-label-group">
                  <span className={`badge ${prediction.category.toLowerCase().replace(' ', '-')}`}>
                    {prediction.category}
                  </span>
                  <p className="explanation">
                    Patient shows a <strong>{prediction.category.toLowerCase()}</strong> for chronic cardiovascular events within a 10-year window.
                  </p>
                </div>
              </div>

              {/* Explanations Bar Charts */}
              <div className="explanations-section">
                <h3>Impact Factor Breakdown</h3>
                <div className="factors-list">
                  {prediction.factors.map((factor, idx) => (
                    <div key={idx} className="factor-item">
                      <div className="factor-info">
                        <span className="factor-name">{factor.name}</span>
                        <span className={`impact-badge ${factor.impact.toLowerCase()}`}>{factor.impact}</span>
                      </div>
                      <div className="bar-track">
                        <div 
                          className={`bar-fill ${factor.impact.toLowerCase()}`}
                          style={{ width: `${factor.weight}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations */}
              <div className="recommendations-section">
                <h3>Recommended Action Guidelines</h3>
                <ul>
                  {prediction.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
