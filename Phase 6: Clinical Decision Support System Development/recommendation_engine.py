"""
recommendation_engine.py
Clinical Recommendation Engine
Rule-based generation of possible conditions, recommended diagnostic tests,
and treatment considerations, built on top of risk_assessment.py.
"""

from typing import Dict, Any, List

try:
    from risk_assessment import assess_risk
except ModuleNotFoundError:
    from Files.risk_assessment import assess_risk

_SYMPTOM_RULES = [
    {
        "keywords": ["chest pain", "chest tightness", "radiating to left arm", "pressure in chest"],
        "possible_conditions": ["Acute coronary syndrome should be considered"],
        "recommended_tests": ["ECG", "Cardiac biomarkers (troponin)"],
        "treatment_considerations": ["Continuous cardiac monitoring"],
    },
    {
        "keywords": ["shortness of breath", "difficulty breathing", "dyspnea", "can't breathe"],
        "possible_conditions": [
            "Acute respiratory distress or pulmonary etiology should be considered"
        ],
        "recommended_tests": ["Chest X-ray", "Pulse oximetry trend", "Arterial blood gas"],
        "treatment_considerations": ["Supplemental oxygen as clinically indicated"],
    },
    {
        "keywords": ["fever", "chills", "sepsis", "infection"],
        "possible_conditions": ["Infectious or septic process should be considered"],
        "recommended_tests": ["Complete blood count", "Blood cultures", "Lactate level"],
        "treatment_considerations": ["Consider sepsis protocol per institutional guidelines"],
    },
    {
        "keywords": ["severe headache", "worst headache", "sudden headache"],
        "possible_conditions": [
            "Intracranial hemorrhage or neurological emergency should be considered"
        ],
        "recommended_tests": ["CT head (non-contrast)", "Neurological exam"],
        "treatment_considerations": ["Urgent neurology consultation may be warranted"],
    },
    {
        "keywords": ["abdominal pain", "stomach pain"],
        "possible_conditions": ["Acute abdominal process should be considered"],
        "recommended_tests": ["Abdominal exam", "Basic metabolic panel", "Lipase"],
        "treatment_considerations": ["Surgical consultation if peritoneal signs present"],
    },
]

def _match_symptom_rules(narrative_text: str) -> Dict[str, List[str]]:
    """Scan narrative text for known keyword patterns and aggregate matched suggestions."""
    text = (narrative_text or "").lower()

    possible_conditions: List[str] = []
    recommended_tests: List[str] = []
    treatment_considerations: List[str] = []

    for rule in _SYMPTOM_RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            possible_conditions.extend(rule["possible_conditions"])
            recommended_tests.extend(rule["recommended_tests"])
            treatment_considerations.extend(rule["treatment_considerations"])

    def dedupe(items):
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return {
        "possible_conditions": dedupe(possible_conditions),
        "recommended_tests": dedupe(recommended_tests),
        "treatment_considerations": dedupe(treatment_considerations),
    }

def generate_recommendations(clinical_input: Dict[str, Any]) -> Dict[str, Any]:
    """Combines risk assessment with symptom-based recommendations."""
    risk = assess_risk(clinical_input)

    narrative = clinical_input.get("narrative") or clinical_input.get("symptoms") or ""
    matched = _match_symptom_rules(narrative)

    possible_conditions = matched["possible_conditions"]
    recommended_tests = matched["recommended_tests"]
    treatment_considerations = matched["treatment_considerations"]

    if not possible_conditions:
        possible_conditions = [
            "No specific rule-based pattern matched; clinician review recommended."
        ]

    if risk["risk_level"] == "High" and "Immediate clinician assessment" not in treatment_considerations:
        treatment_considerations.append("Immediate clinician assessment")

    return {
        "risk_level": risk["risk_level"],
        "possible_conditions": possible_conditions,
        "recommended_tests": recommended_tests,
        "treatment_considerations": treatment_considerations,
        "warnings": risk["warnings"],
        "requires_human_review": risk["requires_human_review"],
    }
