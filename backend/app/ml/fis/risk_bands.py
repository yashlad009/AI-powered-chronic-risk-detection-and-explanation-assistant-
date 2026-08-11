"""
Fuzzy Inference System (FIS).
Implements fuzzy rules to categorize predictions into clinical risk bands (low, medium, high).
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class RiskFuzzyInference:
    """Fuzzy Inference System to map quantitative ANN risk scores to qualitative bands."""
    
    def __init__(self):
        # 1. Define antecedents (inputs) and consequents (outputs)
        self.risk_prob = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'risk_prob')
        self.risk_band = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'risk_band')

        # 2. Define membership functions
        # For input (risk_prob): Low, Medium, High
        self.risk_prob['low'] = fuzz.trimf(self.risk_prob.universe, [0, 0, 0.55])
        self.risk_prob['medium'] = fuzz.trimf(self.risk_prob.universe, [0.35, 0.55, 0.75])
        self.risk_prob['high'] = fuzz.trimf(self.risk_prob.universe, [0.55, 1, 1])

        # For output (risk_band) scale:
        self.risk_band['low'] = fuzz.trimf(self.risk_band.universe, [0, 0, 5])
        self.risk_band['medium'] = fuzz.trimf(self.risk_band.universe, [3, 5, 7])
        self.risk_band['high'] = fuzz.trimf(self.risk_band.universe, [5, 10, 10])

        # 3. Define rules
        rule1 = ctrl.Rule(self.risk_prob['low'], self.risk_band['low'])
        rule2 = ctrl.Rule(self.risk_prob['medium'], self.risk_band['medium'])
        rule3 = ctrl.Rule(self.risk_prob['high'], self.risk_band['high'])

        # 4. Control system creation
        self.control_sys = ctrl.ControlSystem([rule1, rule2, rule3])
        self.simulator = ctrl.ControlSystemSimulation(self.control_sys)

    def infer(self, prob: float) -> dict:
        """
        Takes a raw probability (0 to 1) and computes the qualitative risk band and confidence.
        
        Args:
            prob (float): Raw risk probability from ANN model.
            
        Returns:
            dict: {
                "band": str,        # "Low", "Medium", or "High"
                "confidence": float # Membership degree of the defuzzified value in the selected band
            }
        """
        # Constrain input to [0.0, 1.0] to handle edge cases
        prob_constrained = max(0.0, min(1.0, float(prob)))
        
        try:
            self.simulator.input['risk_prob'] = prob_constrained
            self.simulator.compute()
            output_val = self.simulator.output['risk_band']
            
            # Map the defuzzified value (0 to 10) to Low/Medium/High bands
            low_m = fuzz.interp_membership(self.risk_band.universe, self.risk_band['low'].mf, output_val)
            med_m = fuzz.interp_membership(self.risk_band.universe, self.risk_band['medium'].mf, output_val)
            high_m = fuzz.interp_membership(self.risk_band.universe, self.risk_band['high'].mf, output_val)
            
            memberships = {"Low": low_m, "Medium": med_m, "High": high_m}
            band = max(memberships, key=memberships.get)
            
            return {
                "band": band,
                "confidence": round(float(memberships[band]), 4)
            }
        except Exception as e:
            # Fallback in case of any computation issues
            if prob_constrained < 0.35:
                return {"band": "Low", "confidence": 1.0}
            elif prob_constrained < 0.65:
                return {"band": "Medium", "confidence": 1.0}
            else:
                return {"band": "High", "confidence": 1.0}
