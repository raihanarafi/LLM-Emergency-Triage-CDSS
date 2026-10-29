"""
cdss.py — Master Clinical Decision Support System Integration.
Integrates:
- Clinical Input Data
- Phase 5: Triage Prediction (ESI & Confidence)
- Rule-based Risk/Recommendation Engine
- Qwen LLM Reasoning Engine

Produces a unified, standardized CDSS response schema.
"""

import os
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CDSS_Integration")

# Dynamic sub-module imports
try:
    from clinical_input import get_clinical_input
except ImportError:
    logger.warning("clinical_input module not found. Direct calls must pass clinical_input dicts.")
    get_clinical_input = None

try:
    from triage_predict import predict_triage
except ImportError:
    logger.warning("triage_predict module not found. Using fallback triage logic.")
    predict_triage = None

try:
    from cdss_rules import run_cdss_rules
except ImportError:
    logger.warning("cdss_rules module not found. Using fallback rules logic.")
    run_cdss_rules = None

try:
    from cdss_llm import generate_cdss_llm
except ImportError:
    logger.warning("cdss_llm module not found. Using fallback LLM mock logic.")
    generate_cdss_llm = None


def merge_list_field(llm_list: list, rule_list: list) -> list:
    """Combines list items from LLM and Rule Engine while removing duplicates and preserving order."""
    combined = []
    seen = set()

    for item in (llm_list or []) + (rule_list or []):
        if not item or not isinstance(item, str):
            continue
        cleaned = item.strip()
        cleaned_lower = cleaned.lower()
        if cleaned_lower not in seen:
            seen.add(cleaned_lower)
            combined.append(cleaned)

    return combined


def determine_highest_risk(llm_risk: str, rule_risk: str) -> str:
    """Selects the most conservative (highest priority) risk level between LLM and Rule Engine."""
    risk_hierarchy = {"Critical": 4, "High": 3, "Moderate": 2, "Low": 1, "Unknown": 0}

    l_score = risk_hierarchy.get(str(llm_risk).capitalize(), 0)
    r_score = risk_hierarchy.get(str(rule_risk).capitalize(), 0)

    highest_score = max(l_score, r_score)
    for risk_name, score in risk_hierarchy.items():
        if score == highest_score and score > 0:
            return risk_name

    return "Unknown"


def run_cdss_pipeline(patient_id: int, clinical_input: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Executes the complete end-to-end CDSS integration pipeline.

    Args:
        patient_id (int): Identifier for the patient.
        clinical_input (dict, optional): Clinical payload. If None, retrieved via Person A loader.

    Returns:
        dict: Standardized CDSS output.
    """
    logger.info(f"Executing CDSS integration pipeline for Patient ID: {patient_id}")

    # 1. Fetch Clinical Input (Person A)
    if clinical_input is None:
        if get_clinical_input is not None:
            try:
                clinical_input = get_clinical_input(patient_id)
            except Exception as e:
                logger.error(f"Failed to fetch clinical input for patient {patient_id}: {e}")
                clinical_input = {}
        else:
            clinical_input = {}

    if "patient" not in clinical_input:
        clinical_input["patient"] = {"patient_id": patient_id}
    else:
        clinical_input["patient"]["patient_id"] = patient_id

    # 2. Extract or Predict Triage (Phase 5)
    triage_info = clinical_input.get("triage", {})
    esi = triage_info.get("esi")
    confidence = triage_info.get("confidence", 0.0)

    if (esi is None or confidence == 0.0) and predict_triage is not None:
        try:
            triage_res = predict_triage(clinical_input)
            esi = triage_res.get("esi", esi)
            confidence = triage_res.get("confidence", confidence)
        except Exception as e:
            logger.error(f"Phase 5 Triage prediction error: {e}")

    if esi is None:
        esi = 3
        confidence = 0.50
        logger.warning(f"Defaulting ESI to 3 (Confidence: 0.50) for patient {patient_id}")

    clinical_input["triage"] = {"esi": esi, "confidence": confidence}

    # 3. Execute Rule Engine (Person B)
    rule_output = {}
    if run_cdss_rules is not None:
        try:
            rule_output = run_cdss_rules(clinical_input)
        except Exception as e:
            logger.error(f"Person B Rule Engine execution error: {e}")

    # 4. Execute Qwen LLM Reasoning (Person C)
    llm_output = {}
    if generate_cdss_llm is not None:
        try:
            llm_output = generate_cdss_llm(clinical_input)
        except Exception as e:
            logger.error(f"Person C LLM Reasoning Engine execution error: {e}")
            llm_output = {
                "risk_level": "Unknown",
                "possible_conditions": [],
                "recommended_tests": [],
                "treatment_considerations": [],
                "clinical_reasoning": f"LLM execution failed: {str(e)}",
                "warnings": ["LLM service unavailable. Rule-based assessment applied."],
                "requires_human_review": True
            }

    # 5. Merge Outputs & Determine Conservative Risk Level
    final_risk = determine_highest_risk(
        llm_output.get("risk_level", "Unknown"),
        rule_output.get("risk_level", "Unknown")
    )

    combined_conditions = merge_list_field(
        llm_output.get("possible_conditions", []),
        rule_output.get("possible_conditions", [])
    )

    combined_tests = merge_list_field(
        llm_output.get("recommended_tests", []),
        rule_output.get("recommended_tests", [])
    )

    combined_treatments = merge_list_field(
        llm_output.get("treatment_considerations", []),
        rule_output.get("treatment_considerations", [])
    )

    combined_warnings = merge_list_field(
        llm_output.get("warnings", []),
        rule_output.get("warnings", [])
    )

    # Human Review Safety Rules: ESI 1 or 2, Critical/High Risk, or explicit module flag
    requires_review = (
        bool(llm_output.get("requires_human_review", False)) or
        bool(rule_output.get("requires_human_review", False)) or
        (esi in [1, 2]) or
        (final_risk in ["High", "Critical"])
    )

    reasoning = llm_output.get("clinical_reasoning", "")
    if not reasoning and rule_output.get("rule_summary"):
        reasoning = rule_output.get("rule_summary")

    # Build standardized output schema
    final_response = {
        "patient_id": patient_id,
        "triage": {
            "esi": int(esi),
            "confidence": float(round(confidence, 2))
        },
        "clinical_decision_support": {
            "risk_level": final_risk,
            "possible_conditions": combined_conditions,
            "recommended_tests": combined_tests,
            "treatment_considerations": combined_treatments,
            "clinical_reasoning": reasoning,
            "warnings": combined_warnings,
            "requires_human_review": requires_review
        }
    }

    return final_response
