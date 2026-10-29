"""
cdss_llm_api.py — FastAPI wrapper around cdss_llm.py (Phase 6).
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from cdss_llm import generate_cdss_llm

app = FastAPI(title="CDSS LLM Reasoning Service")


class Patient(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None


class Triage(BaseModel):
    esi: Optional[int] = None
    confidence: Optional[float] = None


class ClinicalInputRequest(BaseModel):
    patient: Patient
    vitals: Dict[str, Any] = {}
    symptoms: str = ""
    medical_history: List[str] = []
    triage: Triage


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/cdss/llm")
def cdss_llm_endpoint(request: ClinicalInputRequest):
    clinical_input = request.dict()
    return generate_cdss_llm(clinical_input)
