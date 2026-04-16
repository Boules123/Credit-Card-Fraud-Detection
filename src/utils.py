"""
Utility functions for hyperparameter tuning and model evaluation.

Supports three search strategies:
  1. grid search      
  2. random search
"""

import time
import pandas as pd

from sklearn.metrics import (
    f1_score, precision_score, 
    recall_score, roc_auc_score, 
    classification_report, confusion_matrix)

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from config import CV_FOLDS, SCORING_METRIC, RANDOM_STATE


#  ---- Hyperparameter tuning ----
def hyperparameter_tuning(
    model,
    param_grid,
    X_train,
    y_train,
    *,
    search_method="grid",
    cv_folds=CV_FOLDS,
    scoring=SCORING_METRIC,
    n_iter=50,
    verbose=1,
):
    """
    Tune hyperparameters with GridSearchCV or RandomizedSearchCV.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    if search_method == "random": # random
        searcher = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=verbose,
            refit=True,
        )
    else:  # grid
        searcher = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=verbose,
            refit=True,
        )

    print(f"\n{'='*60}")
    print(f"Starting tuning using {search_method} search...")
    print(f"Estimator: {type(model).__name__}")
    print(f"Scoring: {scoring}")
    print(f"CV folds: {cv_folds}")
    if search_method == "random":
        print(f"Iterations: {n_iter}")
    total_combos = _count_combinations(param_grid) if search_method == "grid" else n_iter
    print(f"Total fits: {total_combos * cv_folds}")
    print(f"{'='*60}")

    start = time.time()
    searcher.fit(X_train, y_train)
    elapsed = time.time() - start

    
    print(f"\nTuning completed in {elapsed:.1f}s")
    print(f"Best {scoring}: {searcher.best_score_:.4f}")
    print(f"Best params:")
    for k, v in searcher.best_params_.items():
        print(f"{k}: {v}")

    # Build a tidy results table (top 10)
    cv_results_df = pd.DataFrame(searcher.cv_results_)
    top_results = (
        cv_results_df[["params", f"mean_test_score", f"std_test_score", "rank_test_score"]]
        .sort_values("rank_test_score")
        .head(10)
    )
    print(f"\n  Top 10 configurations:")
    print(top_results.to_string(index=False))
    print(f"{'='*60}\n")

    return searcher.best_estimator_, searcher.best_params_, cv_results_df


def _count_combinations(param_grid: dict) -> int:
    """Count total combinations in a parameter grid."""
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count


def compare_models(all_metrics: dict):
    """Print a summary table comparing all trained models."""
    print("\n" + "=" * 70)
    print(f"{'Model':<25s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} {'ROC-AUC':>9s}")
    print("-" * 70)
    for name, metrics in all_metrics.items():
        print(
            f"{name:<25s} "
            f"{metrics.get('f1', float('nan')):>8.4f} "
            f"{metrics.get('precision', float('nan')):>10.4f} "
            f"{metrics.get('recall', float('nan')):>8.4f} "
            f"{metrics.get('roc_auc', float('nan')):>9.4f}"
        )
    print("=" * 70)

    best_model = max(all_metrics, key=lambda m: all_metrics[m].get("f1", 0))
    print(f"\nBest model by F1: {best_model} "
          f"(F1 = {all_metrics[best_model]['f1']:.4f})")

    return best_model


# ---- Evaluation metrics and reporting ----
def evaluate_model(y_true, y_pred, y_scores=None):
    """
    Print classification report, confusion matrix, and key metrics.
    If y_scores (predicted probabilities) are provided, also compute ROC‑AUC.
    Returns a dict of metric values for programmatic use.
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    if y_scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
    else:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    for name, value in metrics.items():
        print(f"{name:>12s}: {value:.4f}")

    return metrics
