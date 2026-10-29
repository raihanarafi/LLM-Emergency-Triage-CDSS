# Phase 6: Clinical Decision Support System (CDSS) Development

## 📌 Phase Overview
Phase 6 combines deterministic rule-based clinical scoring engines with fine-tuned Large Language Model (Qwen2.5-3B) natural language reasoning to build an interactive, safety-guarded **Clinical Decision Support System (CDSS)** for Emergency Departments.

The system consumes patient demographics, vital signs, chief complaints, medical history, and **Phase 5 ESI triage predictions** to generate structured, actionable clinical recommendations—including risk level stratification, differential condition considerations, diagnostic test suggestions, safety warnings, and explanatory clinical reasoning.

> **Disclaimer:** This module is strictly designed for clinical decision support and **does not replace human clinical judgment**. All system outputs carry mandatory human-in-the-loop review flags (`"requires_human_review": true`).

---

## 🛠️ Architecture & Workflow

```text
                       ┌─────────────────────────┐
                       │    Patient Database     │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Person A: Data Builder  │
                       │   (clinical_input.py)   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Phase 5 Triage Engine  │
                       │     (ESI Levels 1–5)    │
                       └────────────┬────────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│ Person B: Risk Engine   │                   │ Person C: Qwen LLM      │
│  (risk_assessment.py &  │                   │     (cdss_llm.py)       │
│ recommendation_engine)  │                   │                         │
└────────────┬────────────┘                   └────────────┬────────────┘
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Person D: Integrator   │
                       │        (cdss.py)        │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │    Final CDSS Output    │
                       └─────────────────────────┘
