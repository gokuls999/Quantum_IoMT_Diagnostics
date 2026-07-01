# Quantum-Enhanced IoMT Diagnostic Framework

**PhD Research — Binu**

A hierarchical IoMT architecture using quantum computing principles to enhance diagnostic accuracy, computation efficiency, and reliability of health monitoring.

---

## Research Problem

Current IoMT systems suffer from:
- Noisy biosignals
- Feature redundancy
- High computational cost
- Poor diagnosis accuracy → ineffective clinical decisions

Existing research treats signal processing, optimization, and diagnosis **separately**. This framework is the **first to combine quantum learning + adaptive optimization + explainable AI in a single integrated IoMT pipeline**.

---

## Pipeline Architecture (9 Stages)

```
INPUT  — Multimodal Physiological Data
         ECG · BP · Temperature · EEG · Pulse Oximeter

  Stage 1  ANUKF          Adaptive Neural Unscented Kalman Filter
           → Biosignal denoising & uncertainty handling

  Stage 2  Q-Flex ViT     Quantum Flexibility Vision Transformer
           → Multimodal feature extraction (quantum-hybrid ViT)

  Stage 3  BMOCO          Binary Multi-Objective Cheetah Optimization
           → Optimal feature subset selection

  Stage 4  HQAN           Hybrid Quantum Attention Network
           → Quantum-enhanced attention-based analysis

  Stage 5  DQA + DRA      Dynamic Quantum Attention + Resource Adaptation
           → Adaptive quantum attention weighting & runtime resource management

  Stage 6  RBWKA          Revamped Black-Winged Kite Algorithm
           → Nature-inspired hyperparameter optimization (VQC weights)

  Stage 7  VQC            Variational Quantum Circuits
           → Quantum-classical hybrid final diagnosis prediction

  Stage 8  Adaptive SHARP Adaptive SHAP
           → Clinically interpretable explanations

OUTPUT — Real-time clinical alerts & monitoring
```

---

## Novel Algorithms

| Acronym | Full Name | Role |
|---------|-----------|------|
| ANUKF | Adaptive Neural Unscented Kalman Filter | Biosignal denoising |
| Q-Flex ViT | Quantum Flexibility Vision Transformer | Feature extraction |
| BMOCO | Binary Multi-Objective Cheetah Optimization | Feature selection |
| HQAN | Hybrid Quantum Attention Network | Quantum attention analysis |
| DQA | Dynamic Quantum Attention | Adaptive attention weighting |
| DRA | Dynamic Resource Adaptation | Runtime resource management |
| RBWKA | Revamped Black-Winged Kite Algorithm | VQC weight optimization |
| VQC | Variational Quantum Circuits | Quantum-classical prediction |
| Adaptive SHARP | Adaptive SHAP | Clinical XAI visualization |

---

## Research Gaps Addressed

| Gap | Solution |
|-----|----------|
| Noisy biosignals | ANUKF adaptive non-linear filtering |
| Feature redundancy | BMOCO multi-objective selection |
| Computational cost | DRA dynamic resource allocation + quantum speedup |
| Poor diagnosis accuracy | VQC + HQAN quantum-enhanced inference |
| Lack of explainability | Adaptive SHARP clinical visualization |
| Isolated research silos | Single integrated pipeline (signal → feature → predict → explain) |

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `pennylane>=0.36.0`, `pennylane-lightning>=0.36.0`, `streamlit>=1.37.0`

---

## Run

```bash
streamlit run app.py --server.port 8504
```

Open `http://localhost:8504`

---

## File Structure

```
├── app.py                — 9-page Streamlit dashboard
├── pipeline.py           — End-to-end pipeline orchestrator
├── quantum_circuits.py   — PennyLane VQC (4-qubit) + Q-Flex attention circuit + HQAN
├── anukf.py              — Adaptive Neural Unscented Kalman Filter
├── bmoco.py              — Binary Multi-Objective Cheetah Optimization
├── rbwka.py              — Revamped Black-Winged Kite Algorithm
├── data_generator.py     — Synthetic multimodal biosignal generation
├── hospital_bridge.py    — Optional: live MediCore HMS data integration
└── requirements.txt
```

---

## Dashboard Pages

1. Pipeline Overview — Architecture diagram, research gaps, KPIs
2. Biosignal Input — Synthetic ECG/BP/Temp/EEG/SpO₂ patient data
3. ANUKF Preprocessing — Signal filtering visualization
4. Q-Flex ViT — Feature extraction & quantum circuit diagram
5. BMOCO — Feature selection convergence
6. HQAN + VQC — Quantum prediction, confusion matrix, metrics
7. Adaptive SHARP — Feature importance & explainability
8. Clinical Alerts — Patient risk stratification
9. Hospital Comparison — Quantum vs traditional monitoring side-by-side
