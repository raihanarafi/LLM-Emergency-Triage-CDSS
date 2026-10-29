'''
clinical_input.py
Patient Data & Clinical Input Processing
'''

import os
import json
import pandas as pd

try:
    from triage_module import predict_triage, FEATURE_COLUMNS
except ModuleNotFoundError:
    from Files.triage_module import predict_triage, FEATURE_COLUMNS


def get_clinical_input(patient_id: int, csv_path: str = None) -> dict:
    """
    Retrieves patient data, handles missing fields, fetches the triage score,
    and standardizes the output for the CDSS.
    """
    if csv_path is None:
        possible_paths = [
            "Files/preprocessed_dataset.csv",
            "preprocessed_dataset.csv",
            "../Files/preprocessed_dataset.csv",
            "/content/drive/MyDrive/MIMIC-IV-Ext Triage Instruction Corpus/Final_Files/Phase_6_Final/Files/preprocessed_dataset.csv",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break

    if not csv_path or not os.path.exists(csv_path):
        return {"error": "Database file 'preprocessed_dataset.csv' not found."}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to read database: {str(e)}"}

    if patient_id < 0 or patient_id >= len(df):
        return {"error": f"Patient ID {patient_id} not found. Range: 0 to {len(df)-1}"}

    row = df.iloc[patient_id]

    def safe_get(val, default="Not available"):
        return val if pd.notnull(val) else default

    narrative_text = str(row.get("triagecase", row.get("chiefcomplaint", "No narrative available.")))

    structured_feats = {col: row[col] for col in FEATURE_COLUMNS if col in row}
    triage_result = predict_triage(structured_feats, narrative_text)

    cdss_input = {
        "patient_id": int(patient_id),
        "patient": {
            "age": safe_get(row.get("age")),
            "gender": "Male" if row.get("gender_M") == 1.0 else "Female",
        },
        "symptoms": narrative_text,
        "vitals": {
            "heart_rate": safe_get(row.get("heartrate")),
            "sbp": safe_get(row.get("sbp")),
            "dbp": safe_get(row.get("dbp")),
            "spo2": safe_get(row.get("o2sat")),
            "temperature": safe_get(row.get("temperature")),
        },
        "medical_history": [],
        "allergies": [],
        "medications": [],
        "laboratory_results": [],
        "narrative": narrative_text,
        "triage": {
            "esi": triage_result.get("predicted_esi", 0),
            "confidence": triage_result.get("confidence", 0.0),
        },
    }

    return cdss_input
