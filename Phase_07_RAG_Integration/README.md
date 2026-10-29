# Phase 7: Retrieval-Augmented Generation (RAG) Integration

## 📌 Executive Summary & Objectives
Phase 7 integrates **Retrieval-Augmented Generation (RAG)** into the Clinical Decision Support System (CDSS). By dynamically supplying our fine-tuned LLM with exact, real-time contextual passages from official medical reference books, the system guarantees grounded clinical recommendations, eliminates hallucinations, and provides verifiable page-level source citations.

### Reference Textbooks
1. **AHRQ Emergency Severity Index (ESI) Implementation Handbook (v4):** Authoritative guide for emergency triage prioritization (Phase 5).
2. **Tintinalli’s Emergency Medicine Manual:** Standard medical reference for differential diagnoses, lab recommendations, and emergency treatment plans (Phase 6).

---

## 🎯 Key Architectural Benefits
- **Ground Truth Medical Data:** System responses are strictly constrained to retrieved textbook passages.
- **Hallucination Elimination:** Prevents critical clinical errors, such as miscalculated drug dosages or inappropriate triage priority levels.
- **Auditability & Citations:** Every recommendation generated for clinicians explicitly states the book title, chapter, section, and page number.
- **Dual-Source Routing:**
  - *ESI Handbook* ➔ Answers *"How urgently does this patient require care?"*
  - *Tintinalli’s Manual* ➔ Answers *"What condition might this be, and how do we evaluate and treat it?"*

---

## 🛠️ End-to-End System Workflow

```text
               User Input (Patient Vitals, Symptoms, History)
                                    │
                                    ▼
                         Query Router 
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   Triage Query (ESI)                             CDSS Query (Tintinalli)
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                     Hybrid Retrieval Engine 
                   (Dense Vectors + Sparse BM25 Search)
                                    │
                                    ▼
                     Re-Ranker Model (BGE / Cohere)
                       (Top 3–5 Context Snippets)
                                    │
                                    ▼
                      Prompt Engineering 
                     (Strict Context-Bound Prompts)
                                    │
                                    ▼
                       Fine-Tuned LLM Generation
                                    │
                                    ▼
             Structured Clinical Output + Direct Page Citations
