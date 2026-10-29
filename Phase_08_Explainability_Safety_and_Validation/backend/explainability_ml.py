import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List

class MLExplainabilityEngine:
    """
    SHAP-based explainability module for structured XGBoost triage models.
    """
    
    def __init__(self, model_path: str, feature_names: List[str] = None):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = feature_names

    def get_patient_explanation(self, patient_features: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates SHAP attributions for a single patient record.
        """
        df_single = pd.DataFrame([patient_features])
        
        # Enforce exact feature columns and order expected by XGBoost
        if self.feature_names:
            for col in self.feature_names:
                if col not in df_single.columns:
                    df_single[col] = 0.0
            df_single = df_single[self.feature_names]
            
        # Select numeric types only (convert booleans to float)
        df_single = df_single.select_dtypes(include=['number', 'bool'])
        df_single = df_single.astype(float)
            
        shap_values = self.explainer.shap_values(df_single)
        
        # Handle Multi-class vs Single-output XGBoost SHAP values
        if isinstance(shap_values, list):
            pred_class = int(self.model.predict(df_single)[0])
            patient_shap = shap_values[pred_class][0]
        elif len(shap_values.shape) == 3:
            pred_class = int(self.model.predict(df_single)[0])
            patient_shap = shap_values[0, :, pred_class]
        else:
            patient_shap = shap_values[0]

        feature_impacts = []
        for feat_name, val, impact in zip(df_single.columns, df_single.iloc[0], patient_shap):
            impact_direction = "High Risk" if impact > 0 else "Low Risk / Protective"
            feature_impacts.append({
                "feature": str(feat_name),
                "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                "impact_score": float(impact),
                "impact": f"{impact_direction} ({impact:+.2f})"
            })

        # Sort by absolute SHAP magnitude descending
        feature_impacts.sort(key=lambda x: abs(x['impact_score']), reverse=True)
        
        results = []
        for item in feature_impacts[:top_k]:
            results.append({
                "feature": item["feature"],
                "value": item["value"],
                "impact": item["impact"]
            })
            
        return results
