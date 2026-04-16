# Credit Card Fraud Detection Pipeline

Production-style machine learning pipeline for binary fraud detection on highly imbalanced transactional data.

This project focuses on practical model development workflows used in real teams:
- clean modular code structure
- reproducible training and evaluation
- class imbalance handling
- hyperparameter tuning
- model comparison and artifact persistence

## Table of Contents

- [Credit Card Fraud Detection Pipeline](#credit-card-fraud-detection-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Business Goal](#business-goal)
  - [Key Capabilities](#key-capabilities)
  - [Project Structure](#project-structure)
  - [System Workflow](#system-workflow)
  - [Supported Models](#supported-models)
  - [Evaluation Strategy](#evaluation-strategy)
  - [Getting Started](#getting-started)
    - [1) Prerequisites](#1-prerequisites)
    - [2) Install dependencies](#2-install-dependencies)
    - [3) Prepare data](#3-prepare-data)
  - [Configuration](#configuration)
  - [How to Train](#how-to-train)
    - [Train one model](#train-one-model)
    - [Train one model with tuning](#train-one-model-with-tuning)
    - [Train all models and compare](#train-all-models-and-compare)
    - [Train all models + tuning + CV + plots](#train-all-models--tuning--cv--plots)
  - [Inference API](#inference-api)
  - [Artifacts and Outputs](#artifacts-and-outputs)
  - [Operational Notes](#operational-notes)
  - [Troubleshooting](#troubleshooting)
    - [Missing model file](#missing-model-file)
    - [Imbalanced metrics look unstable](#imbalanced-metrics-look-unstable)
    - [Plot windows do not appear](#plot-windows-do-not-appear)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)

## Overview

The pipeline trains and evaluates multiple machine learning models for fraud detection, then persists selected artifacts for reuse.

Primary script:
- `src/training.py`

Core modules:
- `src/data_helper.py` for loading, splitting, scaling, resampling, and plotting
- `src/model.py` for model factories and persistence helpers
- `src/utils.py` for tuning, model comparison, and metric reporting
- `src/inference.py` for model scoring on provided test tensors
- `src/config.py` for project configuration constants

## Business Goal

Fraud detection usually suffers from severe class imbalance where fraudulent transactions are rare.
This project optimizes for practical detection quality using:
- imbalance-aware training (SMOTE or random oversampling)
- model-level class weighting where relevant
- threshold-free ranking quality via ROC-AUC
- business-friendly classification metrics (F1, precision, recall)

## Key Capabilities

- Multi-model training from a single CLI entrypoint
- Optional hyperparameter search using grid or randomized search
- Cross-validation support for stability checks
- Validation and held-out test style evaluation inside the training flow
- Confusion matrix and ROC curve plotting
- Saved model and scaler artifacts for downstream inference

## Project Structure

```text
project_2 & 3/
	README.md
	requirements.txt
	input/
	logs/
	models/
	notebooks/
	src/
		config.py
		data_helper.py
		data.py
		inference.py
		model.py
		training.py
		utils.py
		neural_network/
			nn_torch.py
```

## System Workflow

1. Load dataset from configured location.
2. Split into train, validation, and test subsets.
3. Resample train only (SMOTE or random oversampling).
4. Fit scaler on train only, transform validation/test.
5. Train one model or all registered models.
6. Optionally run hyperparameter tuning.
7. Evaluate on validation and then test.
8. Persist model and scaler artifacts.
9. Optionally run cross-validation diagnostics.

## Supported Models

The model registry in `src/model.py` currently supports:

- logistic_regression
- decision_tree
- random_forest
- gradient_boosting
- adaboost
- svm
- knn
- voting

## Evaluation Strategy

The pipeline reports:
- F1 score
- Precision
- Recall
- ROC-AUC
- Classification report
- Confusion matrix

Why these metrics matter:
- F1 balances missed fraud and false alarms.
- Precision controls alert fatigue for analysts.
- Recall tracks fraud capture effectiveness.
- ROC-AUC evaluates ranking quality independent of one fixed threshold.

## Getting Started

### 1) Prerequisites

- Python 3.10 or newer (3.11 recommended)
- pip

### 2) Install dependencies

From project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Prepare data

By default, configuration expects files under `data/`:

- `data/train.csv`
- `data/trainval.csv`
- `data/val.csv`
- `data/test.csv`

Current training flow uses `trainval.csv` and performs internal train/val/test splitting.

Minimum schema requirement:
- target column name: `Class`
- binary values in `Class`: 0 (legit), 1 (fraud)

## Configuration

Adjust runtime behavior in `src/config.py`:

- `RESAMPLING_METHOD`: `smote` or `random`
- `CV_FOLDS`: cross-validation folds
- `SCORING_METRIC`: optimization metric (default `f1`)
- `TUNING_SEARCH_METHOD`: `grid` or `random`
- `TUNING_N_ITER`: randomized search iterations
- `RANDOM_STATE`: reproducibility seed

## How to Train

Run from project root.

### Train one model

```bash
python src/training.py --model random_forest
```

### Train one model with tuning

```bash
python src/training.py --model random_forest --tune --search grid
```

### Train all models and compare

```bash
python src/training.py --all
```

### Train all models + tuning + CV + plots

```bash
python src/training.py --all --tune --search random --cv --plot
```

## Inference API

Inference is exposed as a callable function in `src/inference.py` and is used by the training pipeline.

Example programmatic usage:

```python
import joblib
from inference import run_inference

scaler = joblib.load("models/scaler.joblib")
metrics = run_inference(model_name="random_forest", scaler=scaler, X_test=X_test, y_test=y_test, plot=False)
print(metrics)
```

## Artifacts and Outputs

Generated outputs:

- `models/<model_name>.joblib` trained model artifact
- `models/scaler.joblib` fitted scaler
- `logs/` runtime logs directory (project-level)

Recommendation for production hardening:
- version model artifacts with timestamp and git commit hash
- store metrics snapshots for experiment tracking
- add data drift checks before inference

## Operational Notes

- Resampling is applied to train split only to avoid leakage.
- Scaling is fit on train split only.
- The code is deterministic under the configured random seed for sklearn components.
- For large datasets, prefer randomized search over full grid search.

## Troubleshooting

### Missing model file

If inference raises file-not-found for model artifacts, train first:

```bash
python src/training.py --model random_forest
```

### Imbalanced metrics look unstable

- Try `--tune --search random`
- Compare multiple models with `--all`
- Use `--cv` for fold-level stability insight

### Plot windows do not appear

Use an interactive Python environment or disable plotting with no `--plot` flag.

## Roadmap

- Add proper train/val/test artifact logging per run
- Add experiment tracking (MLflow or Weights and Biases)
- Add unit tests and data contract tests
- Add model calibration and threshold policy configuration
- Add Docker support and CI pipeline

## Contributing

1. Create a feature branch.
2. Keep changes scoped and reproducible.
3. Validate training flow locally.
4. Open a PR with before/after metrics and rationale.

---

If you want, I can also generate a professional CONTRIBUTING.md and a reproducible Makefile for common tasks (setup, train, tune, evaluate).
