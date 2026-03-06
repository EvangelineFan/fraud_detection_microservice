# Fraud Detection System – XGBoost + Production Simulation

## 🎥 Video Walkthrough

A full walkthrough of the project, including feature engineering decisions, model training, and how production considerations are factored into ML system design:

▶️ **YouTube:** [Watch the full walkthrough](https://youtu.be/RtydKrwOhsM?si=dVnywMQM5aqQuXjo)

---

## Overview

This project implements an end-to-end fraud detection pipeline using the PaySim synthetic transaction dataset.  

It includes:

- Feature engineering (including velocity-based behavioral features)
- Time-based data splitting (train / validation / production)
- Model training using XGBoost
- Threshold selection targeting ≥99% recall
- Model artifact generation
- Production simulation-ready outputs

The goal is to demonstrate production-oriented ML workflow, not just model accuracy.

---

## Dataset

- Synthetic financial transaction data (PaySim)
- Highly imbalanced fraud classification problem
- Fraud rate ≈ low (realistic banking scenario)

---

## Project Structure

├── artifacts/ # Saved model + metadata
├── data/ # Local datasets (ignored in Git)
├── venv/ # Virtual environment (ignored)
├── fraud_detection_base_model.ipynb
├── requirements.txt
├── README.md
└── .gitignore

---

## Modeling Approach

### 1. Feature Engineering
- Native transaction features
- Behavioral velocity features (rolling statistics)
- Removed raw time index from model inputs to prevent simulation leakage

### 2. Time-Based Split
- Train → Early time window
- Validation → Later time window
- Production → Future time window (simulated deployment)

### 3. Model
- XGBoost classifier
- `scale_pos_weight` used for class imbalance
- Evaluated using:
  - ROC-AUC
  - PR-AUC (primary metric)

### 4. Threshold Selection
Threshold selected on validation set to achieve:

> ≥ 99% Recall

Performance then verified on future production window.

---

## Model Selection & Diagnostics

Initial experiments were conducted with LightGBM due to its strong performance in tabular data.

While ROC-AUC was high, the model produced a coarse probability distribution due to leaf-wise growth and histogram binning. This resulted in limited threshold granularity for high-recall operating points.

XGBoost was therefore selected as the final model due to:
- Smoother probability calibration
- More stable precision–recall behavior
- Better threshold control for high-recall deployment requirements

Additionally, benchmarking against native transactional features showed that PaySim’s rule-based synthetic fraud structure already provides strong signal. Velocity features were retained to demonstrate production-style behavioral engineering, though gains were modest due to dataset restrictions.

---

## Results

- Validation PR-AUC: ~0.90
- Production Recall (≥99% target): ~0.999
- Production Precision: ~0.77

Strong ranking performance with high recall operating point.

---

## Artifacts Generated

- `model.json`
- `threshold.json`
- `feature_schema.json`
- `metadata.json`

These artifacts are structured for deployment readiness (e.g., Vertex AI).

---

## How to Run

1. Create virtual environment
2. Install dependencies: pip install -r requirements.txt
3. Run notebook: fraud_detection_base_model.ipynb

---

## Notes

- PaySim fraud is rule-based synthetic data.
- Velocity features demonstrate production-style feature engineering.
- Threshold selection reflects real-world fraud system constraints (high recall requirement).

---

## Future Work

- Model deployment on Vertex AI
- Streaming simulation
- Drift monitoring
- Automated retraining pipeline