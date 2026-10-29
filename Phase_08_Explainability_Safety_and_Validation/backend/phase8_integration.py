"""
phase8_integration.py — Phase 8 master integration module.

evaluate_patient_full(...) runs ONE patient through every piece built in Phases 5-8
and returns one standardized dict:

    triage (ML + LLM)  ->  CDSS reasoning  ->  SHAP explainability
    ->  vitals guardrails (on RAW vitals)  ->  RAG citations  ->  LLM safety checks
"""

import sys
import os
import math
import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. AUTO-REGISTER TEAMMATE BACKEND PATHS (LOCAL DYNAMIC RESOLUTION)
# ----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to Phase_08_Explainability_Safety_and_Validation directory
BASE_PHASE8 = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

team_paths = [
    CURRENT_DIR,               # Phase_08.../backend
    BASE_PHASE8,               # Phase_08...
]

for path in team_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

# Import teammates' modules
from explainability_ml import MLExplainabilityEngine          # Person 1
from vitals_guardrails import VitalsGuardrails                # Person 1
from rag_validation import RAGRetrievalValidator              # Person 2
from citation_verifier import CitationVerifier                # Person 2
from llm_guardrails import check_llm_safety                   # Person 3
from llm_explainability import extract_structured_rationale   # Person 3
from llm_explainability import evaluate_confidence_trigger    # Person 3

FEATURES = [
    "age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain",
    "gender_M", "arrival_transport_UNKNOWN", "arrival_transport_WALK IN",
]
CONTINUOUS = ["age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]


# ----------------------------------------------------------------------------
# Small Helpers
# ----------------------------------------------------------------------------
def clean(v):
    """numpy -> plain python, NaN -> None (so JSON / API calls never break)."""
    if isinstance(v, np.generic):
        v = v.item()
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def recover_scaler(raw_df, std_df, key="tiragecase"):
    """
    preprocessed_dataset.csv is the raw file after z-score scaling: z = (x - mean) / std.
    That is a straight line, so a line fit on matching rows recovers mean and std exactly.
    Returns (params, report). report[f] shows how well the line fits (should be ~0).
    """
    raw_u = raw_df.drop_duplicates(key, keep=False)
    std_u = std_df.drop_duplicates(key, keep=False)[[key] + CONTINUOUS]
    pairs = raw_u.merge(std_u, on=key, suffixes=("_raw", "_std"))
    params, report = {}, {}
    for f in CONTINUOUS:
        x = pairs[f + "_raw"].astype(float)
        z = pairs[f + "_std"].astype(float)
        ok = x.notna() & z.notna()
        if ok.sum() < 3 or x[ok].nunique() < 2:
            report[f] = "not enough data to recover"
            continue
        slope, intercept = np.polyfit(x[ok], z[ok], 1)
        err = float(np.max(np.abs(z[ok] - (slope * x[ok] + intercept))))
        params[f] = {"mean": float(-intercept / slope), "std": float(1.0 / slope)}
        report[f] = f"max fit error {err:.5f} on {int(ok.sum())} rows"
    return params, report


def standardize_raw(raw, scaler_params):
    """Raw clinical values -> the 11 standardized model features. Returns (features, notes)."""
    std, notes = {}, []
    for f in CONTINUOUS:
        val_key = f
        if f == "o2sat" and raw.get("o2sat") is None:
            val_key = "spo2"
            
        x = clean(raw.get(val_key))
        if x is None or f not in scaler_params:
            std[f] = 0.0
            notes.append(f"{f} missing - treated as average for the ML model")
        else:
            std[f] = (float(x) - scaler_params[f]["mean"]) / scaler_params[f]["std"]
            
    std["gender_M"] = 1.0 if str(raw.get("gender", "")).strip().upper().startswith("M") else 0.0
    transport = str(raw.get("arrival_transport", "")).strip().upper()
    std["arrival_transport_UNKNOWN"] = 1.0 if transport == "UNKNOWN" else 0.0
    std["arrival_transport_WALK IN"] = 1.0 if transport == "WALK IN" else 0.0
    return std, notes


# ----------------------------------------------------------------------------
# Master Pipeline Class
# ----------------------------------------------------------------------------
class Phase8Pipeline:
    def __init__(self, ml_model_path, predict_triage_llm, generate_cdss_llm,
                 phase7_url, scaler_params=None):
        self.ml_model = joblib.load(ml_model_path)
        self.ml_explainer = MLExplainabilityEngine(model_path=ml_model_path, feature_names=FEATURES)
        self.predict_triage_llm = predict_triage_llm      # Phase 5 function
        self.generate_cdss_llm = generate_cdss_llm        # Phase 6 function
        self.rag = RAGRetrievalValidator(api_url=phase7_url.rstrip("/") + "/search")  # Phase 7 API
        self.verifier = CitationVerifier()
        self.scaler_params = scaler_params

    # ---- step 1: triage ------------------------------------------------------
    def _ml_predict(self, std):
        X = pd.DataFrame([{f: float(std[f]) for f in FEATURES}])[FEATURES]
        esi = int(self.ml_model.predict(X)[0]) + 1        # model classes are 0-4 -> ESI 1-5
        confidence = float(np.max(self.ml_model.predict_proba(X)[0]))
        return esi, confidence

    def _llm_predict(self, narrative):
        try:
            return self.predict_triage_llm(narrative)
        except Exception as e:
            return {"predicted_esi": None, "confidence": 0.0, "valid": False,
                    "reason": f"LLM triage failed: {e}"}

    # ---- step 2: CDSS reasoning ----------------------------------------------
    def _cdss(self, raw, narrative, esi, esi_confidence):
        spo2_val = clean(raw.get("o2sat")) if clean(raw.get("o2sat")) is not None else clean(raw.get("spo2"))
        
        vitals = {
            "heart_rate": clean(raw.get("heartrate")) or clean(raw.get("hr")),
            "sbp": clean(raw.get("sbp")),
            "dbp": clean(raw.get("dbp")),
            "spo2": spo2_val,
            "temperature": clean(raw.get("temperature")) or clean(raw.get("temp")),
            "resp_rate": clean(raw.get("resprate")) or clean(raw.get("rr")),
            "pain": clean(raw.get("pain")),
        }
        vitals = {k: v for k, v in vitals.items() if v is not None}
        age = clean(raw.get("age"))
        clinical_input = {
            "patient": {"age": int(age) if age is not None else None, "gender": clean(raw.get("gender"))},
            "vitals": vitals,
            "symptoms": narrative,
            "medical_history": [],
            "triage": {"esi": int(esi), "confidence": float(esi_confidence)},
        }
        try:
            return self.generate_cdss_llm(clinical_input)
        except Exception as e:
            return {"risk_level": "Unknown", "possible_conditions": [], "recommended_tests": [],
                    "treatment_considerations": [], "warnings": [f"CDSS failed: {e}"],
                    "clinical_reasoning": "", "requires_human_review": True,
                    "valid": False, "raw_output": ""}

    # ---- step 3: RAG ---------------------------------------------------------
    def _retrieve(self, raw, narrative):
        complaint = clean(raw.get("chiefcomplaint")) or narrative[:200]
        plan = [("ESI_Handbook", str(complaint)),
                ("Tintinalli_Manual", f"{complaint} emergency evaluation and management")]
        citations, passages = [], []
        for source, query in plan:
            for hit in self.rag.query_phase7_api(query, source_filter=source, top_k=2):
                citations.append({
                    "source": hit.get("source"), "chapter": hit.get("chapter"),
                    "section_title": hit.get("section_title"), "page": hit.get("page_number"),
                    "snippet": (hit.get("matched_snippet") or "")[:300], "score": hit.get("score"),
                })
                passages.append({"source": hit.get("source", ""),
                                 "text": hit.get("context_text") or hit.get("matched_snippet") or ""})
        return citations, passages

    # ---- the master function -------------------------------------------------
    def evaluate_patient_full(self, patient_id, raw_patient_data, standardized_features, clinical_narrative):
        raw = raw_patient_data
        warnings, reasons = [], []

        # 1) triage: ML + LLM, most urgent wins
        ml_esi, ml_conf = self._ml_predict(standardized_features)
        llm = self._llm_predict(clinical_narrative)
        llm_esi = llm.get("predicted_esi") if llm.get("valid") else None
        llm_conf = float(llm.get("confidence") or 0.0)
        _, low_conf = evaluate_confidence_trigger([llm_conf], threshold=0.70)
        agreement = bool(llm_esi is not None and llm_esi == ml_esi)
        final_esi = min(e for e in (ml_esi, llm_esi) if e is not None)
        if agreement:
            confidence = (ml_conf + llm_conf) / 2
        else:
            confidence = min(ml_conf, llm_conf) if llm_esi is not None else ml_conf

        # 2) vitals guardrails on RAW values
        vitals_override, vitals_warnings = VitalsGuardrails.evaluate_vitals({k: clean(v) for k, v in raw.items()})
        if vitals_override and final_esi > 2:
            warnings.append(f"Vitals look critical but models predicted ESI {final_esi} - "
                            "clinician should consider ESI 2 or more urgent.")

        # 3) SHAP explanation
        try:
            top_features = self.ml_explainer.get_patient_explanation(standardized_features, top_k=5)
        except Exception as e:
            top_features = []
            warnings.append(f"SHAP explanation unavailable: {e}")

        # 4) CDSS reasoning (Phase 6) - receives the final ESI
        cdss = self._cdss(raw, clinical_narrative, final_esi, confidence)
        reasoning = cdss.get("clinical_reasoning") or llm.get("reason") or ""

        # 5) LLM safety checks (Person 3)
        scan_text = " ".join([str(cdss.get("raw_output", "")), str(reasoning),
                              " ".join(map(str, cdss.get("treatment_considerations", []))),
                              str(llm.get("reason", ""))])
        safety = check_llm_safety(scan_text)
        rationale = extract_structured_rationale(str(cdss.get("raw_output", "")))

        # 6) RAG citations + faithfulness (Person 2)
        citations, passages = self._retrieve(raw, clinical_narrative)
        grounding = self.verifier.verify_response_grounding(reasoning, passages)

        # 7) should a human review this case?
        if vitals_override:
            reasons.append("vitals guardrail triggered")
        if final_esi in (1, 2):
            reasons.append(f"high acuity (ESI {final_esi})")
        if llm_esi is None:
            reasons.append("LLM triage prediction missing or unreadable")
        elif not agreement:
            reasons.append(f"ML (ESI {ml_esi}) and LLM (ESI {llm_esi}) disagree")
        if low_conf:
            reasons.append(f"LLM confidence below 0.70 ({llm_conf:.2f})")
        if cdss.get("requires_human_review", True):
            reasons.append("CDSS asked for human review")
        if not cdss.get("valid", False):
            reasons.append("CDSS output could not be parsed")
        if safety["llm_guardrail_status"] != "PASSED":
            reasons.append("LLM safety guardrail flagged the text")
        if not citations:
            reasons.append("no reference citations retrieved")

        return {
            "patient_id": clean(patient_id),
            "triage_summary": {
                "predicted_esi": int(final_esi),
                "ml_prediction": ml_esi,
                "llm_prediction": llm_esi,
                "agreement": agreement,
                "confidence": round(float(confidence), 4),
                "ml_confidence": round(ml_conf, 4),
                "llm_confidence": round(llm_conf, 4),
                "human_review_required": bool(reasons),
                "human_review_reasons": reasons,
            },
            "clinical_decision_support": {
                "risk_level": cdss.get("risk_level"),
                "possible_conditions": cdss.get("possible_conditions", []),
                "recommended_tests": cdss.get("recommended_tests", []),
                "treatment_considerations": cdss.get("treatment_considerations", []),
                "warnings": list(cdss.get("warnings", [])) + warnings,
            },
            "explainability": {
                "ml_top_features": top_features,
                "llm_reasoning": reasoning,
                "llm_triage_reason": llm.get("reason"),
                "key_symptoms_identified": rationale["key_symptoms_identified"],
                "contraindications_or_warnings": list(cdss.get("warnings", [])) + rationale["contraindications_or_warnings"],
                "rag_citations": citations,
            },
            "safety_and_guardrails": {
                "vitals_override_triggered": bool(vitals_override),
                "vitals_warnings": vitals_warnings,
                "faithfulness_score": grounding["grounding_overlap_score"],
                "is_fully_grounded": grounding["is_fully_grounded"],
                "inline_citations_found": len(grounding["extracted_citations"]),
                "llm_guardrail_status": safety["llm_guardrail_status"],
                "llm_guardrail_violations": safety["violations"],
            },
        }

    # ---- same thing, but starting from RAW values (used by the API) ----------
    def evaluate_from_raw(self, patient_id, raw, clinical_narrative):
        if not self.scaler_params:
            raise ValueError("scaler_params are needed to standardize raw vitals for the ML model")
        std, notes = standardize_raw(raw, self.scaler_params)
        out = self.evaluate_patient_full(patient_id, raw, std, clinical_narrative)
        out["clinical_decision_support"]["warnings"].extend(notes)
        return out
