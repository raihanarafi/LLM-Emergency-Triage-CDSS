'''
clinical_input_api.py
FastAPI Interface for Clinical Data Retrieval
'''

from fastapi import APIRouter, HTTPException
from clinical_input import get_clinical_input

router = APIRouter()

@router.get("/api/clinical-input/{patient_id}")
def fetch_patient_data(patient_id: int):
    """
    API endpoint to retrieve the standardized CDSS input for a specific patient.
    """
    data = get_clinical_input(patient_id)

    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])

    return data
