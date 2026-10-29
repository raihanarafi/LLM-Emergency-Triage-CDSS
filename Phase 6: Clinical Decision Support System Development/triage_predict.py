import joblib
import pandas as pd
from llm_triage_inference import predict_triage_llm

# Loading person A model
ML_MODEL_PATH = "triage_ml_model.pkl"
ml_model = joblib.load("/content/drive/MyDrive/MIMIC-IV-Ext Triage Instruction Corpus/Final_Files/Phase_6_Final/Files/triage_ml_model.pkl")

# Selected features required by Person A's XGBoost model
FEATURE_COLUMNS = [
    "age", "temperature", "heartrate", "resprate", "o2sat",
    "sbp", "dbp", "pain", "gender_M",
    "arrival_transport_UNKNOWN", "arrival_transport_WALK IN"
]

def predict_triage(structured_data: dict, narrative_text: str) -> dict:
    """
    Predicts patient ESI level using both Machine Learning and LLM predictions.
    
    Parameters:
        structured_data (dict): Feature dictionary matching FEATURE_COLUMNS.
        narrative_text (str): Patient clinical narrative text.
        
    Returns:
        dict: Standardized output format matching Phase 5 specification.
    """
    # 1. Machine Learning Prediction (Person A)
    df_feat = pd.DataFrame([structured_data])[FEATURE_COLUMNS]
    raw_ml_pred = ml_model.predict(df_feat)[0]
    ml_pred = int(raw_ml_pred + 1)  # Convert XGBoost 0-4 labels back to ESI 1-5
    
    # 2. LLM Narrative Prediction (Person B)
    llm_res = predict_triage_llm(narrative_text)
    llm_pred = llm_res.get("predicted_esi")
    
    # 3. Decision Integration Logic
    if llm_pred is not None and 1 <= llm_pred <= 5:
        agreement = (ml_pred == llm_pred)
        
        if agreement:
            final_esi = ml_pred
            human_review = False
        else:
            # Conservative Strategy: Select lower ESI number (higher urgency)
            final_esi = min(ml_pred, llm_pred)
            human_review = True
            
        confidence = round((0.4 * 0.85 + 0.6 * llm_res.get("confidence", 0.5)), 4)
        reason = llm_res.get("reason", "No narrative explanation provided.")
    else:
        # Fallback if LLM output fails
        final_esi = ml_pred
        agreement = False
        human_review = True
        confidence = 0.5
        reason = "LLM prediction unavailable; defaulting to ML model prediction."

    return {
        "predicted_esi": final_esi,
        "ml_prediction": ml_pred,
        "llm_prediction": llm_pred,
        "agreement": agreement,
        "confidence": confidence,
        "reason": reason,
        "human_review": human_review
    }