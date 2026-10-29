import re

def check_llm_safety(response_text: str) -> dict:
    """
    Scans the LLM output text for unsafe phrases, absolute dosage prescriptions, 
    or forbidden directives to ensure clinical safety compliance.
    """
    if not response_text:
        return {
            "llm_guardrail_status": "PASSED",
            "violations": []
        }

    text_to_scan = str(response_text)

    unsafe_patterns = [
        r"\byou must take\b",
        r"\badminister immediately\b",
        r"\bguarantee\b",
        r"\bdefinitely do not consult a doctor\b",
        r"\btake \d+ mg of\b",
        r"\bdo not go to the hospital\b"
    ]
    
    violations = []
    for pattern in unsafe_patterns:
        if re.search(pattern, text_to_scan, re.IGNORECASE):
            violations.append(f"Matched restricted phrase pattern: {pattern}")
            
    is_safe = len(violations) == 0
    return {
        "llm_guardrail_status": "PASSED" if is_safe else "FLAGGED",
        "violations": violations
    }
