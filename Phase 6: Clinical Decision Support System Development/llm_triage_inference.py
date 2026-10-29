"""
llm_triage_inference.py

LLM-based triage inference module (Person B, Phase 5).
Loads the fine-tuned Qwen2.5-3B-Instruct model + LoRA adapter and exposes
predict_triage_llm(patient_text) -> dict, matching the Phase 5 output spec:

    {
        "predicted_esi": int or None,
        "confidence": float,
        "reason": str,
        "valid": bool,
        "raw_output": str
    }
"""

import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/MIMIC-IV-Ext Triage Instruction Corpus/Final_Files/Phase_6_Final/Files/fine_tuned_qwen/fine_tuned_qwen"

SYSTEM_PROMPT = (
    "You are an AI emergency triage assistant. Given a patient's narrative, respond with "
    "ONLY a JSON object in this exact format, no other text:\n"
    '{"predicted_esi": <integer 1-5>, "reason": "<one sentence clinical justification>"}'
)

_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=_bnb_config,
    device_map={"": "cpu"}
)

model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

def build_prompt(patient_text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_text},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)


def generate_response(patient_text: str, max_new_tokens: int = 120):
    inputs = build_prompt(patient_text)
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids, output.scores


def parse_llm_output(generated_text: str):
    try:
        match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            esi = int(parsed.get("predicted_esi"))
            reason = str(parsed.get("reason", "")).strip()
            if 1 <= esi <= 5:
                return esi, reason, True
    except Exception:
        pass

    esi_match = re.search(r"predicted_esi[\"']?\s*:?\s*(\d)", generated_text)
    reason_match = re.search(r"reason[\"']?\s*:?\s*[\"'](.+?)[\"']", generated_text)
    if esi_match:
        esi = int(esi_match.group(1))
        reason = reason_match.group(1).strip() if reason_match else "Reason not extracted."
        if 1 <= esi <= 5:
            return esi, reason, True

    return None, "Could not parse a valid ESI level from model output.", False


def compute_confidence(generated_ids, scores, predicted_esi):
    if predicted_esi is None:
        return 0.0
    target_str = str(predicted_esi)
    for step, token_id in enumerate(generated_ids):
        token_str = tokenizer.decode([token_id]).strip()
        if token_str == target_str:
            step_scores = scores[step][0]
            probs = torch.softmax(step_scores, dim=-1)
            return round(probs[token_id].item(), 4)
    return 0.5


def predict_triage_llm(patient_text: str):
    generated_text, generated_ids, scores = generate_response(patient_text)
    predicted_esi, reason, is_valid = parse_llm_output(generated_text)

    if not is_valid:
        return {
            "predicted_esi": None,
            "confidence": 0.0,
            "reason": reason,
            "valid": False,
            "raw_output": generated_text,
        }

    confidence = compute_confidence(generated_ids, scores, predicted_esi)
    return {
        "predicted_esi": predicted_esi,
        "confidence": confidence,
        "reason": reason,
        "valid": True,
        "raw_output": generated_text,
    }
