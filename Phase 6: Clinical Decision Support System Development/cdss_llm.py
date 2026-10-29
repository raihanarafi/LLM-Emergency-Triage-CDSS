"""
cdss_llm.py — LLM-based clinical decision support module (Person C, Phase 6).
Exposes generate_cdss_llm(clinical_input: dict) -> dict
"""

import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/MIMIC-IV-Ext Triage Instruction Corpus/Final_Files/Phase_6_Final/Files/fine_tuned_qwen/fine_tuned_qwen"

VALID_RISK_LEVELS = {"Low", "Moderate", "High", "Critical"}

CDSS_SYSTEM_PROMPT = """You are an AI clinical decision support assistant for emergency department staff.
You do NOT diagnose or prescribe treatment. You provide decision-support recommendations only, using
cautious clinical language such as "should be considered", "may be indicated", "consider evaluating for".

Given a patient's clinical information and their ESI (triage) level, respond with ONLY a JSON object in
this exact schema, no other text, no markdown code fences, no comments:

{
  "risk_level": "<Low, Moderate, High, or Critical>",
  "possible_conditions": ["<condition to consider>", ...],
  "recommended_tests": ["<test to consider>", ...],
  "treatment_considerations": ["<consideration>", ...],
  "clinical_reasoning": "<1-3 sentence explanation grounded only in the provided information>",
  "warnings": ["<any urgent safety warning, if applicable>"],
  "requires_human_review": <true or false>
}

Rules:
- Do not invent clinical details not present in the input.
- ESI 1-2 should generally set requires_human_review to true.
- If information is insufficient or contradictory, say so in clinical_reasoning and set requires_human_review to true.
- Output ONLY the JSON object. No explanation before or after it.
"""

# Model loading

_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, quantization_config=_bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)


# Prompt construction

def build_clinical_summary(clinical_input: dict) -> str:
    patient = clinical_input.get("patient", {})
    vitals = clinical_input.get("vitals", {})
    symptoms = clinical_input.get("symptoms", "Not specified")
    history = clinical_input.get("medical_history", [])
    esi = clinical_input.get("triage", {}).get("esi", "Unknown")

    vitals_str = ", ".join(f"{k.replace('_', ' ').upper()}: {v}" for k, v in vitals.items()) or "Not recorded"
    history_str = ", ".join(history) if history else "None reported"

    return f"""Patient: {patient.get('age', 'Unknown')}-year-old {patient.get('gender', 'Unknown')}
Chief complaint / symptoms: {symptoms}
Vitals: {vitals_str}
Medical history: {history_str}
ESI (triage) level: {esi}"""


def build_cdss_prompt(clinical_input: dict):
    summary = build_clinical_summary(clinical_input)
    messages = [
        {"role": "system", "content": CDSS_SYSTEM_PROMPT},
        {"role": "user", "content": summary},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)


# Generation

def generate_cdss_response(clinical_input: dict, max_new_tokens: int = 600) -> str:
    inputs = build_cdss_prompt(clinical_input)
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# Parsing

FIELD_PATTERNS = {
    "risk_level": r'"?risk_level"?\s*:?\s*',
    "possible_conditions": r'(?:\d+\.\s*)?"?possible[_ ]conditions"?\s*:?\s*',
    "recommended_tests": r'(?:\d+\.\s*)?"?recommended[_ ]tests"?\s*:?\s*',
    "treatment_considerations": r'(?:\d+\.\s*)?"?treatment[_ ]considerations"?\s*:?\s*',
    "clinical_reasoning": r'(?:\d+\.\s*)?"?clinical[_ ]reasoning"?\s*:?\s*',
    "warnings": r'(?:\d+\.\s*)?"?warnings"?\s*:?\s*',
    "requires_human_review": r'(?:\d+\.\s*)?"?requires[_ ]human[_ ]review"?\s*:?\s*',
}


def extract_list_field(section_text: str):
    section_text = section_text.strip()

    bracket_match = re.search(r"\[(.*?)\]", section_text, re.DOTALL)
    if bracket_match:
        inner = bracket_match.group(1)
        items = re.findall(r'"([^"]*)"', inner)
        if items:
            return [i.strip() for i in items if i.strip()]
        if inner.strip() == "":
            return []

    if re.search(r'(?:^|\s)-\s+\S', section_text):
        items = re.split(r'\s*-\s+', section_text)
        return [i.strip(' "\'.,') for i in items if i.strip()]

    return []


def extract_scalar_field(section_text: str):
    section_text = section_text.strip()
    quoted = re.search(r'"([^"]*)"', section_text)
    if quoted and len(quoted.group(1)) > 3:
        return quoted.group(1).strip()

    cleaned = re.sub(r'^[:\-\s"\[]+', '', section_text)
    cleaned = re.split(r'\s{2,}|"\s*,', cleaned)[0]
    return cleaned.strip(' "\'.,')


def universal_field_extract(raw_text):
    matches = []
    for field_name, pattern in FIELD_PATTERNS.items():
        for m in re.finditer(pattern, raw_text, re.IGNORECASE):
            matches.append((m.start(), m.end(), field_name))
    if not matches:
        return {}
    matches.sort(key=lambda x: x[0])
    dedup, seen = [], set()
    for start, end, name in matches:
        if name in seen:
            continue
        seen.add(name)
        dedup.append((start, end, name))
    dedup.sort(key=lambda x: x[0])
    sections = {}
    for i, (start, end, name) in enumerate(dedup):
        next_start = dedup[i + 1][0] if i + 1 < len(dedup) else len(raw_text)
        sections[name] = raw_text[end:next_start]
    return sections


def build_result_from_sections(sections):
    risk_level = "Unknown"
    rl_section = sections.get("risk_level", "")
    m = re.search(r"\b(Low|Moderate|High|Critical)\b", rl_section[:60], re.IGNORECASE)
    if m:
        risk_level = m.group(1).capitalize()

    review_text = sections.get("requires_human_review", "")
    requires_human_review = True
    head = review_text.strip()[:30]
    if re.search(r"\b(false|no)\b", head, re.IGNORECASE):
        requires_human_review = False
    elif re.search(r"\b(true|yes)\b", head, re.IGNORECASE):
        requires_human_review = True

    return {
        "risk_level": risk_level,
        "possible_conditions": extract_list_field(sections.get("possible_conditions", "")),
        "recommended_tests": extract_list_field(sections.get("recommended_tests", "")),
        "treatment_considerations": extract_list_field(sections.get("treatment_considerations", "")),
        "clinical_reasoning": extract_scalar_field(sections.get("clinical_reasoning", "")),
        "warnings": extract_list_field(sections.get("warnings", "")),
        "requires_human_review": requires_human_review,
    }


def is_substantive(result: dict) -> bool:
    return bool(
        result["possible_conditions"] or result["recommended_tests"] or len(result["clinical_reasoning"]) > 15
    )


def safe_default():
    return {
        "risk_level": "Unknown", "possible_conditions": [], "recommended_tests": [],
        "treatment_considerations": [],
        "clinical_reasoning": "Automated reasoning unavailable — model output could not be parsed.",
        "warnings": ["LLM output was invalid or unparseable. Manual clinical review required."],
        "requires_human_review": True, "valid": False, "parse_method": "failed",
    }


def parse_cdss_output(raw_text: str) -> dict:
    raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()

    line_stripped = re.sub(r'//[^\n]*', '', raw_text)
    try:
        match = re.search(r"\{.*\}", line_stripped, re.DOTALL)
        candidate = match.group(0) if match else None
        if not candidate and line_stripped.count('{') > 0:
            candidate = line_stripped[line_stripped.index('{'):].rstrip()
            candidate = re.sub(r',\s*$', '', candidate)
            candidate += ']' * max(candidate.count('[') - candidate.count(']'), 0)
            candidate += '}' * max(candidate.count('{') - candidate.count('}'), 0)
        if candidate:
            parsed = json.loads(candidate)
            risk_level = str(parsed.get("risk_level", "Unknown")).strip()
            if risk_level not in VALID_RISK_LEVELS:
                risk_level = "Unknown"
            result = {
                "risk_level": risk_level,
                "possible_conditions": list(parsed.get("possible_conditions", [])),
                "recommended_tests": list(parsed.get("recommended_tests", [])),
                "treatment_considerations": list(parsed.get("treatment_considerations", [])),
                "clinical_reasoning": str(parsed.get("clinical_reasoning", "")).strip(),
                "warnings": list(parsed.get("warnings", [])),
                "requires_human_review": bool(parsed.get("requires_human_review", True)),
            }
            if is_substantive(result):
                result["valid"] = True
                result["parse_method"] = "json"
                return result
    except Exception:
        pass

    comment_preserved = raw_text.replace("//", "  ")
    sections = universal_field_extract(comment_preserved)
    if sections:
        result = build_result_from_sections(sections)
        if is_substantive(result):
            result["valid"] = True
            result["parse_method"] = "fallback_universal"
            return result

    return safe_default()


# Safety overrides

def apply_safety_overrides(result: dict, clinical_input: dict) -> dict:
    esi = clinical_input.get("triage", {}).get("esi")

    if esi in (1, 2):
        result["requires_human_review"] = True

    if esi in (1, 2) and result["risk_level"] in ("Low", "Moderate"):
        result["warnings"].append(
            f"Model risk assessment ({result['risk_level']}) is lower than expected for ESI {esi}. Verify manually."
        )
        result["requires_human_review"] = True

    return result


# Full pipeline

def generate_cdss_llm(clinical_input: dict) -> dict:
    raw_output = generate_cdss_response(clinical_input)
    result = parse_cdss_output(raw_output)
    result = apply_safety_overrides(result, clinical_input)
    result["raw_output"] = raw_output
    return result
