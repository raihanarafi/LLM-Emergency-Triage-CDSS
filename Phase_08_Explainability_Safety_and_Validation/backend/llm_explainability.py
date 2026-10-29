import json

def extract_structured_rationale(raw_llm_output: str) -> dict:
    """
    Parses Qwen's output text into structured clinical components:
    clinical_justification, key_symptoms_identified, and contraindications_or_warnings.
    """
    structured_data = {
        "clinical_justification": "",
        "key_symptoms_identified": [],
        "contraindications_or_warnings": []
    }

    if not raw_llm_output:
        return structured_data

    text_to_parse = str(raw_llm_output)

    try:
        if "{" in text_to_parse and "}" in text_to_parse:
            json_start = text_to_parse.find("{")
            json_end = text_to_parse.rfind("}") + 1
            parsed_json = json.loads(text_to_parse[json_start:json_end])

            # Fallback mappings for alternative key names from CDSS output
            justification = (
                parsed_json.get("clinical_justification") or 
                parsed_json.get("clinical_reasoning") or 
                text_to_parse
            )
            symptoms = (
                parsed_json.get("key_symptoms_identified") or 
                parsed_json.get("possible_conditions") or 
                []
            )
            warnings = (
                parsed_json.get("contraindications_or_warnings") or 
                parsed_json.get("warnings") or 
                []
            )

            structured_data["clinical_justification"] = justification
            structured_data["key_symptoms_identified"] = symptoms if isinstance(symptoms, list) else [str(symptoms)]
            structured_data["contraindications_or_warnings"] = warnings if isinstance(warnings, list) else [str(warnings)]
            return structured_data
    except Exception:
        pass

    structured_data["clinical_justification"] = text_to_parse.strip()
    return structured_data


def evaluate_confidence_trigger(token_probabilities: list, threshold: float = 0.70) -> tuple:
    """
    Parses token probabilities from the generated ESI digit. 
    Returns a tuple of (average_confidence_score, human_review_required_boolean).
    """
    if not token_probabilities:
        return 0.0, True

    valid_probs = [float(p) for p in token_probabilities if p is not None]
    if not valid_probs:
        return 0.0, True

    avg_confidence = sum(valid_probs) / len(valid_probs)
    human_review_required = avg_confidence < threshold

    return round(avg_confidence, 4), human_review_required
