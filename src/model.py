"""
Model definitions for Credit Card Fraud Detection.

Each builder returns a configured (but untrained) estimator.
`get_model()` maps a string name to the corresponding builder.
"""

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from config import RANDOM_STATE


# logistic regression
def build_logistic_regression(use_grid=False):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    param_grid = None
    if use_grid:
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }
    return model, param_grid


# decision tree
def build_decision_tree(use_grid=False):
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    param_grid = None
    if use_grid:
        param_grid = {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    return model, param_grid

# random forest
def build_random_forest(use_grid=False):
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_grid = None
    if use_grid:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    return model, param_grid

# gradient boosting
def build_gradient_boosting(use_grid=False):
    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )
    param_grid = None
    if use_grid:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    return model, param_grid


# adaboost
def build_adaboost(use_grid=False):
    model = AdaBoostClassifier(
        random_state=RANDOM_STATE,
    )
    param_grid = None
    if use_grid:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
        }
    return model, param_grid

# support vector machine
def build_svm(use_grid=False):
    model = SVC(
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )
    param_grid = None
    if use_grid:
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto"] + list(np.logspace(-3, 2, 6)),
        }
    return model, param_grid

# k-nearest neighbors
def build_KNN(use_grid=False):
    model = KNeighborsClassifier()
    param_grid = None
    if use_grid:
        param_grid = {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        }
    return model, param_grid

# voting classifier (ensemble of logistic regression, decision tree, and random forest)
def build_voting_classifier(use_grid=False):
    lr, _ = build_logistic_regression()
    dt, _ = build_decision_tree()
    rf, _ = build_random_forest()
    model = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("dt", dt),
            ("rf", rf),
        ],
        voting="soft",
        n_jobs=-1,
    )
    return model, None  # voting classifier doesn't support grid search directly



# model registry mapping string names to builder functions
_MODEL_REGISTRY = {
    "logistic_regression": build_logistic_regression,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
    "adaboost": build_adaboost,
    "svm": build_svm,
    "knn": build_KNN,
    "voting": build_voting_classifier,
}

# get_model() looks up the builder by name and returns the model instance and param_grid (if any)
def get_model(name, use_grid: bool = False) -> tuple:
    """Return an untrained model instance by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](use_grid)


# prediction and model persistence functions
def predict(model, X):
    return model.predict(X)


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


# save and load model using joblib
def save_model(model, file_path: str):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path: str):
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model
