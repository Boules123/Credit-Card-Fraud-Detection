"""
Data loading, preprocessing, resampling, evaluation, and visualization utilities.

Key design choices
──────────────────
* The scaler is fit **only** on training data to prevent data leakage.
* SMOTE / random oversampling is applied **only** to training data.
* Stratified splitting preserves the fraud‑ratio across splits.
* imblearn.pipeline.Pipeline is used so resamplers work inside a pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Literal, overload

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score,
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline  # supports resamplers, unlike sklearn Pipeline

from config import TARGET_COL, TEST_SIZE, RANDOM_STATE


def load_data(
    file_path,
    split=False,
):
    """Load dataset csv and optionally split into train/val/test sets."""
    df = pd.read_csv(file_path)
    if split:
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            df.drop(TARGET_COL, axis=1), df[TARGET_COL], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_val_test
        )
        return X_train, y_train, X_val, y_val, X_test, y_test
    return df


# visualization functions for data exploration

def visualize_data(df: pd.DataFrame) -> None:
    """Show class distribution, correlation heatmap, and feature histograms."""
    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Feature histograms
    df.drop(TARGET_COL, axis=1).hist(figsize=(20, 20))
    plt.tight_layout()
    plt.show()


# data preprocessing and resampling functions
def separate_features_target(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    return X, y


def scale_data(X_train, X_test=None):
    """
    Fit StandardScaler on X_train only, then transform both splits.
    Returns (X_train_scaled, X_test_scaled, scaler).
    If X_test is None, returns (X_train_scaled, None, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler



def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified train/test split to preserve class ratio."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )



def _get_resampler(method: str):
    """Return a resampler instance based on method name."""
    if method == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    elif method == "random":
        return RandomOverSampler(random_state=RANDOM_STATE)
    else:
        raise ValueError("Invalid resampling method. Choose 'smote' or 'random'.")


def resample_data(X_train, y_train, method: str = "smote"):
    """Apply resampling to the training data only."""    
    resampler = _get_resampler(method)
    resampled = resampler.fit_resample(X_train, y_train)
    X_resampled, y_resampled = resampled[0], resampled[1]
    return X_resampled, y_resampled


# pipeline modelling
def create_pipeline(model, resampling_method: str = "smote"):
    """
    Build an imblearn Pipeline: scale → resample → model.
    Uses imblearn.pipeline.Pipeline so the resampler step is supported.
    """
    resampler = _get_resampler(resampling_method)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("resampler", resampler),
        ("model", model),
    ])
    return pipeline


# visualization functions for evaluation reporting
def plot_roc_curve(y_true, y_scores):
    """Plot the ROC curve given true labels and predicted probabilities."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title):
    """Plot a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

