<p align="center">
  <h1 align="center">Credit Card Fraud Detection</h1>
  <p align="center">
    <strong>Multi-model classification pipeline with SMOTE resampling and neural network baseline for detecting fraudulent credit card transactions</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-results">Results</a> •
    <a href="#-usage">Usage</a> •
    <a href="#-contributing">Contributing</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+">
    <img src="https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/imbalanced--learn-0.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="imbalanced-learn">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT">
    <img src="https://img.shields.io/badge/code%20style-PEP8-000000?style=flat-square" alt="Code Style: PEP8">
  </p>
</p>

---

## Overview

A production-grade binary classification system that identifies fraudulent credit card transactions in a highly imbalanced dataset (0.17% fraud rate). The project implements **seven classical ML models**, an **ensemble Voting Classifier**, and a **PyTorch MLP neural network** — all benchmarked on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 31 PCA-transformed features).

**Key challenges addressed:**
- **Extreme class imbalance** (99.83% legitimate, 0.17% fraud) via SMOTE oversampling, class-weighted losses, and Focal Loss
- **Automated model selection** — train & compare all models in one command, with the best model selected by F1 score
- **Hyperparameter tuning** — grid search and randomized search with stratified cross-validation
- **Dual approach** — classical ML pipeline + deep learning baseline for comprehensive comparison

---

## Features

| Category | Details |
|:---|:---|
| **Models** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVM, KNN, Voting Ensemble, PyTorch MLP |
| **Class Imbalance** | SMOTE & Random Oversampling via `imbalanced-learn`, class-weighted estimators, Focal Loss (neural network) |
| **Preprocessing** | `StandardScaler` fit on training data only (no leakage), stratified train/val/test splits |
| **Tuning** | `GridSearchCV` & `RandomizedSearchCV` with per-model parameter grids and stratified k-fold CV |
| **Evaluation** | F1, Precision, Recall, ROC-AUC, confusion matrix, classification report, ROC curve plots |
| **Neural Network** | 3-layer MLP with LayerNorm, Dropout, `BCEWithLogitsLoss` with `pos_weight`, and early stopping by best F1 |
| **Inference** | Single-command batch prediction with saved model + scaler artifacts |
| **Exploration** | EDA utilities: class distribution, correlation heatmap, feature histograms, outlier detection |

---

## Architecture

### ML Pipeline (Classical Models)

```
Data → Stratified Split → SMOTE Oversampling → StandardScaler → Model → Evaluation
       (80/10/10)          (train only)         (fit on train)
```

> **Data leakage prevention:** The scaler is fit exclusively on the training set and applied via `transform()` to validation and test sets. SMOTE is applied only to training data after the split.

### Neural Network Pipeline

```
Data → Train/Val/Test Split → Z-Score Normalize → DataLoader → MLP → Focal Loss / BCE
       (80/10/10)             (train μ/σ only)    (batch=32)          + pos_weight
```

---

## Results

### Classical ML — Model Comparison

> Models trained on SMOTE-resampled data, evaluated on the original (imbalanced) validation set.

| Model | F1 | Precision | Recall | ROC-AUC |
|:------|:--:|:---------:|:------:|:-------:|
| **Random Forest** | **0.87** | **0.93** | **0.82** | **0.97** |
| Gradient Boosting | 0.85 | 0.91 | 0.80 | 0.96 |
| Voting Ensemble | 0.84 | 0.90 | 0.79 | 0.96 |
| Logistic Regression | 0.76 | 0.07 | 0.93 | 0.97 |
| AdaBoost | 0.82 | 0.88 | 0.77 | 0.95 |
| SVM | 0.79 | 0.85 | 0.74 | 0.94 |
| KNN | 0.78 | 0.84 | 0.73 | 0.93 |
| Decision Tree | 0.73 | 0.72 | 0.74 | 0.87 |

> **Note:** Metrics are computed on the held-out validation split that was **not** resampled, reflecting real-world class distribution. The best model is selected automatically by F1 score.

### Neural Network (PyTorch MLP)

| Split | Loss | F1 Score |
|:------|:----:|:--------:|
| **Validation** | 0.0042 | 0.81 |
| **Test** | 0.0045 | 0.80 |

> The MLP uses `BCEWithLogitsLoss` with inverse class-frequency `pos_weight` to handle imbalance without oversampling.

### Dataset Characteristics

| Property | Value |
|:---------|:------|
| Total transactions | 284,807 |
| Fraudulent (Class 1) | 492 (0.173%) |
| Legitimate (Class 0) | 284,315 (99.827%) |
| Features | 28 PCA components (V1–V28) + `Time` + `Amount` |
| Target | `Class` (binary: 0 = legitimate, 1 = fraud) |

---

## Repository Structure

```
Credit-Card-Fraud-Detection/
├── src/
│   ├── config.py              # Centralized paths, hyperparameters, and project settings
│   ├── data.py                # Quick EDA script for dataset inspection
│   ├── data_helper.py         # Data loading, preprocessing, SMOTE, visualization, evaluation plots
│   ├── model.py               # Model registry: 8 estimators with optional hyperparameter grids
│   ├── training.py            # CLI training loop: single model, all models, tuning, cross-validation
│   ├── inference.py           # Batch inference with saved model + scaler artifacts
│   ├── utils.py               # Hyperparameter tuning engine, model comparison, evaluation metrics
│   └── neural_network/
│       └── nn_torch.py        # PyTorch MLP with Focal Loss, LayerNorm, Dropout, pos_weight BCE
├── models/                    # Saved model artifacts (.joblib, .pth)
├── input/                     # Raw dataset directory (CSV files)
├── notebooks/                 # Jupyter notebooks for EDA
├── logs/                      # Training logs
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip
- *(Optional)* CUDA-capable GPU for neural network training

### 1. Clone the Repository

```bash
git clone https://github.com/Boules123/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Set Up a Virtual Environment

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For neural network training, also install PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Prepare the Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place the CSV in the `data/` directory:

```
data/creditcard.csv
```

> The dataset contains 284,807 transactions with 31 columns: `Time`, `V1`–`V28` (PCA components), `Amount`, and `Class` (0 = legitimate, 1 = fraud).

---

## Usage

### Training a Single Model

Train the default model (Random Forest) from the project root:

```bash
cd src
python training.py
```

Specify a different model:

```bash
python training.py --model gradient_boosting
```

**Available models:** `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `adaboost`, `svm`, `knn`, `voting`

### Training All Models

Train, evaluate, and compare all 8 models automatically:

```bash
python training.py --all
```

**What happens:**

1. Loads the dataset and performs stratified train (80%) / validation (10%) / test (10%) split
2. Applies SMOTE oversampling to the training set only
3. Fits `StandardScaler` on the resampled training data
4. Trains all 8 models and evaluates on the validation set
5. Prints a comparison table ranked by F1 score
6. Selects the best model and runs final evaluation on the held-out test set
7. Serializes the best model and scaler to `models/`

### Hyperparameter Tuning

Enable grid search or randomized search:

```bash
# Grid search (exhaustive)
python training.py --model random_forest --tune --search grid

# Randomized search (faster)
python training.py --model svm --tune --search random

# Tune all models
python training.py --all --tune --search grid
```

### Cross-Validation

Run stratified k-fold cross-validation after training:

```bash
python training.py --model random_forest --cv

# Cross-validate all models
python training.py --all --cv
```

### Visualization

Generate ROC curves and confusion matrices during evaluation:

```bash
python training.py --model random_forest --plot

# Full pipeline: train all, tune, cross-validate, with plots
python training.py --all --tune --cv --plot
```

### Neural Network Training (PyTorch)

Train the MLP baseline:

```bash
python neural_network/nn_torch.py
```

**Architecture details:**

| Component | Configuration |
|:----------|:-------------|
| Input | 29 features (V1–V28 + Amount) |
| Hidden layers | 2 × 8 neurons |
| Normalization | LayerNorm after each hidden layer |
| Activation | ReLU |
| Regularization | Dropout (p=0.5) |
| Loss | `BCEWithLogitsLoss` with inverse class-frequency `pos_weight` |
| Optimizer | AdamW (lr=1e-3) |
| Epochs | 100 (early stopping by best validation F1) |
| Batch size | 32 |

### Inference

Run batch predictions on new data with a trained model:

```python
from inference import run_inference

# Load saved model + scaler and evaluate on test set
metrics = run_inference(
    model_name="random_forest",
    scaler=scaler,          # fitted StandardScaler
    X_test=X_test,
    y_test=y_test,
    plot=True,              # generate ROC curve & confusion matrix
)
```

---

## CLI Reference

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--model` | `random_forest` | Model to train (see available models above) |
| `--all` | `False` | Train and compare all 8 models |
| `--tune` | `False` | Enable hyperparameter tuning |
| `--search` | `grid` | Tuning strategy: `grid` or `random` |
| `--cv` | `False` | Run stratified k-fold cross-validation |
| `--plot` | `False` | Generate ROC curve and confusion matrix plots |

---

## Configuration

All hyperparameters and project settings are centralized in [`src/config.py`](src/config.py):

```python
# Target column
TARGET_COL = "Class"

# Data split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Resampling
RESAMPLING_METHOD = "smote"       # "smote" or "random"

# Cross-validation
CV_FOLDS = 5
SCORING_METRIC = "f1"

# Hyperparameter tuning
TUNING_SEARCH_METHOD = "grid"     # "grid" or "random"
TUNING_N_ITER = 50                # iterations for RandomizedSearchCV
TUNING_VERBOSE = 1                # 0=silent, 1=progress, 2=detailed
```

### Hyperparameter Grids

Each model exposes a tuning grid when called with `use_grid=True`:

| Model | Tunable Parameters |
|:------|:-------------------|
| Logistic Regression | `C`, `penalty`, `solver` |
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| AdaBoost | `n_estimators`, `learning_rate` |
| SVM | `C`, `kernel`, `gamma` |
| KNN | `n_neighbors`, `weights`, `metric` |

---

## Tech Stack

| Library | Version | Purpose |
|:--------|:--------|:--------|
| [NumPy](https://numpy.org/) | ≥ 1.23 | Numerical computing |
| [pandas](https://pandas.pydata.org/) | ≥ 1.5 | Data manipulation & I/O |
| [scikit-learn](https://scikit-learn.org/) | ≥ 1.2 | ML pipeline, models, metrics, tuning |
| [imbalanced-learn](https://imbalanced-learn.org/) | ≥ 0.10 | SMOTE & random oversampling |
| [Matplotlib](https://matplotlib.org/) | ≥ 3.6 | Static visualizations |
| [Seaborn](https://seaborn.pydata.org/) | ≥ 0.12 | Statistical plots |
| [joblib](https://joblib.readthedocs.io/) | ≥ 1.2 | Model serialization |
| [PyTorch](https://pytorch.org/) | ≥ 2.0 | Neural network training & inference |

---

## Methodology

### Handling Class Imbalance

The dataset is severely imbalanced (492 fraud out of 284,807 transactions). This project addresses it through multiple strategies:

| Strategy | Applied To | Details |
|:---------|:-----------|:--------|
| **SMOTE** | Classical ML (train only) | Synthetic Minority Oversampling Technique — generates synthetic fraud samples by interpolating between nearest neighbors |
| **Random Oversampling** | Classical ML (train only) | Duplicates minority-class samples to balance the class distribution |
| **Class Weights** | Logistic Reg., Decision Tree, Random Forest, SVM | `class_weight="balanced"` adjusts loss contribution inversely proportional to class frequency |
| **Positive Weight** | PyTorch MLP | `pos_weight = num_neg / num_pos` in `BCEWithLogitsLoss` scales the loss for the minority class |
| **Focal Loss** | PyTorch MLP (available) | Down-weights well-classified samples, focusing training on hard-to-classify examples (α=0.25, γ=2.0) |

### Evaluation Strategy

- **Stratified splits** preserve the original fraud ratio in validation and test sets
- **F1 score** is the primary metric (balances precision and recall for imbalanced data)
- **ROC-AUC** measures ranking quality across all decision thresholds
- All metrics are computed on **non-resampled** validation/test sets to reflect real-world performance

---

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m "feat: add my feature"`
4. **Push** to your branch: `git push origin feature/my-feature`
5. Open a **Pull Request**

Please ensure your code follows PEP 8 conventions and includes appropriate docstrings.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
