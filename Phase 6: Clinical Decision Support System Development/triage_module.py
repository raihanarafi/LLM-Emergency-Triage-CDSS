"""
triage_module.py
Machine Learning Triage Prediction Module (Phase 5 Model Loader)
"""

import os
import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "age", "heartrate", "sbp", "dbp", "o2sat", "temperature", "respiratory_rate"
]

def _find_model_file():
    possible_paths = [
        "triage_ml_model.pkl",
        "Files/triage_ml_model.pkl",
        "/content/drive/MyDrive/MIMIC-IV-Ext Triage Instruction Corpus/Final_Files/Phase_6_Final/Files/triage_ml_model.pkl",
        "/content/drive/MyDrive/MIMIC-IV-Ext Triage Instruction Corpus/Final_Files/Phase_6/Files/triage_ml_model.pkl",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not locate 'triage_ml_model.pkl' in drive directory.")

_model_cache = None

def load_triage_model():
    global _model_cache
    if _model_cache is None:
        model_path = _find_model_file()
        _model_cache = joblib.load(model_path)
    return _model_cache

def predict_triage(structured_feats: dict, narrative_text: str = "") -> dict:
    """
    Predicts ESI level and confidence score using trained model features.
    """
    try:
        model = load_triage_model()

        # Build single-row DataFrame matching feature order
        input_data = {}
        for col in FEATURE_COLUMNS:
            val = structured_feats.get(col, None)
            if val is None or pd.isna(val) or val == "Not available":
                input_data[col] = [0.0]
            else:
                input_data[col] = [float(val)]

        df_input = pd.DataFrame(input_data)

        # Predict ESI
        pred = model.predict(df_input)[0]

        # Get probability/confidence if model supports it
        confidence = 0.85
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)[0]
            confidence = float(max(probs))

        return {
            "predicted_esi": int(pred),
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        # Fallback response if model fails
        return {
            "predicted_esi": 3,
            "confidence": 0.50,
            "error": str(e)
        }
