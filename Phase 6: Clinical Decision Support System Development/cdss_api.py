"""
cdss_api.py — FastAPI REST Service for Integrated CDSS.
Exposes /cdss/evaluate endpoint combining all.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from cdss import run_cdss_pipeline

app = FastAPI(
    title="Integrated Clinical Decision Support System API",
    description="Person D Integration API combining Clinical Input, Triage, Rules, and LLM Reasoning.",
    version="1.0.0"
)


class PatientSchema(BaseModel):
    patient_id: Optional[int] = 101
    age: Optional[int] = None
    gender: Optional[str] = None


class TriageSchema(BaseModel):
    esi: Optional[int] = None
    confidence: Optional[float] = None


class CDSSRequestSchema(BaseModel):
    patient_id: Optional[int] = 101
    patient: Optional[PatientSchema] = Field(default_factory=PatientSchema)
    vitals: Dict[str, Any] = {}
    symptoms: str = ""
    medical_history: List[str] = []
    triage: Optional[TriageSchema] = Field(default_factory=TriageSchema)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Integrated CDSS Service"}


@app.post("/cdss/evaluate")
def evaluate_cdss_endpoint(request: CDSSRequestSchema):
    try:
        payload = request.dict()
        patient_id = payload.get("patient_id") or payload.get("patient", {}).get("patient_id", 101)

        result = run_cdss_pipeline(patient_id=patient_id, clinical_input=payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CDSS Engine Error: {str(e)}")
