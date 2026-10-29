# Large Language Models for Emergency Triage and Clinical Decision Support

## Project Overview

The objective of this project is to design and develop an Artificial Intelligence (AI)-powered Clinical Decision Support System (CDSS) using Large Language Models (LLMs) to assist healthcare professionals in emergency departments with patient triage, risk assessment, diagnosis support, and treatment recommendations. The system aims to improve the efficiency, accuracy, and consistency of emergency care by analyzing patient symptoms, medical history, laboratory reports, vital signs, and clinical guidelines to provide evidence-based recommendations in real time.

Emergency departments often face challenges such as overcrowding, limited medical staff, delayed decision-making, and high patient volumes. An LLM-based clinical assistant can support physicians and nurses by rapidly processing structured and unstructured clinical information, prioritizing patients based on the severity of their conditions, recommending diagnostic tests, suggesting treatment options, and generating clinical documentation. The system is intended to assist healthcare professionals rather than replace clinical judgment.

The project will integrate Large Language Models, Retrieval-Augmented Generation (RAG), Electronic Health Records (EHRs), clinical guidelines, medical knowledge bases, and explainable AI to provide reliable and transparent clinical decision support.

---

## What We Have to Do

The project can be divided into the following major phases:

- **Phase 1:** Requirement Analysis and System Design
- **Phase 2:** Clinical Data Collection and Knowledge Base Development
- **Phase 3:** Data Preprocessing and Standardization
- **Phase 4:** Large Language Model Selection and Fine-Tuning
- **Phase 5:** Intelligent Emergency Triage Module Development
- **Phase 6:** Clinical Decision Support System Development
- **Phase 7:** Retrieval-Augmented Generation (RAG) Integration
- **Phase 8:** Explainability, Safety, and Validation
- **Phase 9:** System Integration and User Interface Development
- **Phase 10:** Testing, Evaluation, and Clinical Validation
- **Phase 11:** Deployment, Monitoring, and Continuous Learning

---
⚙️ Environment Setup & InstallationHardware & Software Prerequisites
- OS: Linux (Ubuntu 20.04/22.04/24.04) or Windows 10/11 (via WSL2 or Native Python)
- GPU: NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g., RTX 3060, RTX 4060, or workstation equivalent)
- Python: Python 3.10 or 3.11
---

 ## Step 1: Clone Repository & Set Up Virtual Environment
 Open your terminal in the project root directory and create a clean isolated environment:
# Clone the repository
- git clone <YOUR_REPOSITORY_URL>
- cd Phase_9_System Integration and User Interface Development

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

streamlit run app.py
