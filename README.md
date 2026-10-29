# Phase 9: System Integration and User Interface Development
### Large Language Models for Emergency Triage and Clinical Decision Support
**NED University of Engineering & Technology**

---

## 📌 Overview
Phase 9 transitions our backend clinical decision engines—combining Phase 5 XGBoost triage, Phase 4 & 6 fine-tuned Qwen2.5-3B LLM, Phase 7 FAISS Vector RAG, and Phase 8 Explainability & Safety Guardrails—into an interactive, web-based Streamlit User Interface and API layer.

The Streamlit interface acts as a real-time clinical dashboard for emergency clinicians, enabling seamless patient data entry, live local GPU inference, dynamic risk alerts, visual SHAP explainability, clickable RAG source citations, and raw consolidated JSON inspection.

---

## 📁 Repository Directory Structure & File Locations

To run the pipeline and Streamlit UI on a local GPU workstation/laptop, ensure all required model weights, vector database files, and backend scripts are placed in the following project directory layout:

```text
Project_Root/
├── requirements.txt                          # Project dependency specifications
├── app.py                                    # Unified Streamlit Dashboard & In-Process RAG API
├── Phase_04_LLM_Selection_and_Fine_Tuning/
│   └── fine_tuned_qwen/                      # Phase 4 LoRA Adapter Weights & Tokenizer
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
├── Phase_07_RAG_Integration/
│   ├── api/
│   │   └── search_api.py                     # Hybrid Search REST API Service
│   └── vector_db/                            # Chunked Medical Text JSONL Files
│       ├── esi_children.jsonl
│       ├── esi_parents.jsonl
│       ├── tint_children.jsonl
│       └── tint_parents.jsonl
└── Phase_08_Explainability_Safety_and_Validation/
    ├── models/
    │   └── triage_ml_model.pkl              # Saved XGBoost ML Classifier
    └── backend/                              # Phase 8 Pipeline & Guardrail Modules
        ├── phase8_integration.py             # Master Pipeline Engine (Phase8Pipeline)
        ├── explainability_ml.py             # SHAP Attribution Module
        ├── vitals_guardrails.py             # Deterministic Vitals Boundary Net
        ├── rag_validation.py                # Retrieval Precision Evaluator
        ├── citation_verifier.py             # RAG Citation & Faithfulness Checker
        ├── llm_guardrails.py                # Output Safety Filter Rules
        └── llm_explainability.py            # Token Logit Confidence & Rationale Extractor

```
⚙️ Environment Setup & InstallationHardware & Software Prerequisites
- OS: Linux (Ubuntu 20.04/22.04/24.04) or Windows 10/11 (via WSL2 or Native Python)
- GPU: NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g., RTX 3060, RTX 4060, or workstation equivalent)
- Python: Python 3.10 or 3.11
---

 ## Step 1: Clone Repository & Set Up Virtual Environment
 Open your terminal in the project root directory and create a clean isolated environment:Bash# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd Project_Root

# Create a Python virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

## Step 2: Install PyTorch & Project Dependencies
Install CUDA-enabled PyTorch first, then install the package requirements:Bash# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install required project dependencies
pip install -r requirements.txt

🚀 How to Run the Application Locally
Launch Command
In your activated terminal, execute:

Bash
streamlit run app.py
