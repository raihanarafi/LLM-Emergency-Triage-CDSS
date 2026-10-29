"""
risk_assessment.py
Clinical Risk Assessment Engine
Rule-based risk stratification that combines the Phase 5 ESI triage level
with vital-sign red flags to produce a decision-support risk assessment.
"""

from typing import Dict, Any, List

def _to_float(value, default=None):
    """Safely convert a vitals value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _check_vital_red_flags(vitals: Dict[str, Any]) -> List[str]:
    """Inspect vital signs for values outside typical safe ranges."""
    flags = []

    hr = _to_float(vitals.get("heart_rate"))
    sbp = _to_float(vitals.get("sbp"))
    dbp = _to_float(vitals.get("dbp"))
    spo2 = _to_float(vitals.get("spo2"))
    temp = _to_float(vitals.get("temperature"))

    if hr is not None:
        if hr >= 130:
            flags.append("Severe tachycardia detected (HR >= 130 bpm).")
        elif hr >= 100:
            flags.append("Tachycardia detected (HR >= 100 bpm).")
        elif hr <= 50:
            flags.append("Bradycardia detected (HR <= 50 bpm).")

    if sbp is not None:
        if sbp < 90:
            flags.append("Severe hypotension detected (SBP < 90 mmHg).")
        elif sbp >= 180:
            flags.append("Hypertensive urgency range (SBP >= 180 mmHg).")

    if dbp is not None and dbp >= 120:
        flags.append("Severely elevated diastolic pressure (DBP >= 120 mmHg).")

    if spo2 is not None:
        if spo2 < 90:
            flags.append("Severe hypoxia detected (SpO2 < 90%).")
        elif spo2 < 94:
            flags.append("Mild-to-moderate hypoxia detected (SpO2 < 94%).")

    if temp is not None:
        if temp >= 39.0:
            flags.append("High fever detected (Temp >= 39.0C).")
        elif temp <= 35.0:
            flags.append("Hypothermia detected (Temp <= 35.0C).")

    # Shock index (HR / SBP) check
    if hr is not None and sbp not in (None, 0):
        shock_index = hr / sbp
        if shock_index >= 1.0:
            flags.append(
                f"Elevated shock index ({shock_index:.2f}) suggests possible hemodynamic compromise."
            )

    return flags

def _esi_to_baseline_risk(esi) -> str:
    """Map Phase 5 ESI level to a baseline risk category."""
    try:
        esi = int(esi)
    except (TypeError, ValueError):
        return "Unknown"

    if esi in (1, 2):
        return "High"
    if esi == 3:
        return "Moderate"
    if esi in (4, 5):
        return "Low"
    return "Unknown"

_RISK_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Unknown": 1}

def _escalate(risk_a: str, risk_b: str) -> str:
    """Return whichever of the two risk levels is more severe."""
    return risk_a if _RISK_ORDER.get(risk_a, 1) >= _RISK_ORDER.get(risk_b, 1) else risk_b

def assess_risk(clinical_input: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for risk assessment."""
    vitals = clinical_input.get("vitals", {}) or {}
    triage = clinical_input.get("triage", {}) or {}
    esi = triage.get("esi")

    baseline_risk = _esi_to_baseline_risk(esi)
    red_flags = _check_vital_red_flags(vitals)

    severe_keywords = ("Severe", "shock index")
    vital_escalates_to_high = any(
        any(keyword in flag for keyword in severe_keywords) for flag in red_flags
    )

    risk_level = baseline_risk
    if vital_escalates_to_high:
        risk_level = _escalate(risk_level, "High")
    elif red_flags:
        risk_level = _escalate(risk_level, "Moderate")

    warnings = list(red_flags)
    if risk_level == "High":
        warnings.append("Immediate clinician assessment required.")
    if esi in (None, "Unknown") or baseline_risk == "Unknown":
        warnings.append("Triage (ESI) result missing or invalid; risk level may be incomplete.")

    requires_human_review = risk_level in ("High", "Unknown") or bool(red_flags)

    return {
        "risk_level": risk_level,
        "risk_factors": red_flags,
        "warnings": warnings,
        "requires_human_review": requires_human_review,
    }
