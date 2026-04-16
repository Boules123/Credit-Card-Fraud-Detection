"""
Training script for Credit Card Fraud Detection.
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import (
    TRAINVAL_CSV, MODEL_DIR,
    RESAMPLING_METHOD, CV_FOLDS, SCORING_METRIC,
    AVAILABLE_MODELS, RANDOM_STATE,
    TUNING_SEARCH_METHOD, TUNING_N_ITER, TUNING_VERBOSE,
)

from data_helper import (
    load_data, scale_data, resample_data,
    plot_roc_curve, plot_confusion_matrix,
)

from utils import hyperparameter_tuning, compare_models, evaluate_model
from model import get_model, predict, predict_proba, save_model

def val_model(model, X_val, y_val):
    """Evaluate a trained model and return predictions + metrics."""
    y_pred = predict(model, X_val)
    y_scores = predict_proba(model, X_val)
    metrics = evaluate_model(y_val, y_pred, y_scores)
    return y_pred, y_scores, metrics


def do_tuning(
    model,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    search_method,
    model_name,
    *,
    plot: bool = False,
):
    if param_grid:
        # Hyperparameter tuning 
        model, _, _ = hyperparameter_tuning(
            model, param_grid, X_train, y_train,
            search_method=search_method,
            cv_folds=CV_FOLDS,
            scoring=SCORING_METRIC,
            n_iter=TUNING_N_ITER,
            verbose=TUNING_VERBOSE,
        )
    else:
        # Some models (e.g., voting) do not expose a tuning grid.
        model.fit(X_train, y_train)

    print("\n[Test set evaluation]")
    y_pred, y_scores, metrics = val_model(model, X_test, y_test)

    if plot:
        plot_roc_curve(y_test, y_scores)
        plot_confusion_matrix(y_test, y_pred, title=f"{model_name} — Confusion Matrix")

    return model, metrics


def train_model(
    model_name: str,
    X_train, y_train,
    X_test, y_test,
    *,
    tune: bool = False,
    search_method: str = TUNING_SEARCH_METHOD,
    plot: bool = False,
    use_pca: bool = False,
):
    """
    Train - tune (optionally) a single model, then evaluate on test data.

    if tune==True:
      1. Build the model 'and' its param_grid via get_model(use_grid=True).
      2. Run hyperparameter search.
    else:
        1. Build the model with default params via get_model(use_grid=False).
        2. Fit the model on the entire training set.
    """
    print(f"\n{'─'*60}")
    print(f"  Model: {model_name}   |   Tune: {tune}")
    print(f"{'─'*60}")

    model, param_grid = get_model(model_name, use_grid=tune)

    if tune:
        model, test_metrics = do_tuning(
            model,
            param_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            search_method,
            model_name,
            plot=plot,
        )
        return model, test_metrics

    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    model.fit(X_train, y_train)


    # ---- evaluate on train set ----
    print("\n[Train set evaluation]")
    _, _, _ = val_model(model, X_train, y_train)

    # ---- evaluate on test set ----
    print("\n[Test set evaluation]")
    y_pred, y_scores, test_metrics = val_model(model, X_test, y_test)

    if plot:
        plot_roc_curve(y_test, y_scores)
        plot_confusion_matrix(y_test, y_pred, title=f"{model_name} — Confusion Matrix")

    return model, test_metrics


def cross_validate_model(model_name: str, X, y):
    """Perform cross-validation and print results."""
    model, _ = get_model(model_name)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=SCORING_METRIC, n_jobs=-1)
    mean_score = np.mean(cv_results['test_score']) # why? to summarize overall performance across folds
    std_score = np.std(cv_results['test_score']) # why? to show variability across folds
    print(f"  {model_name:25s}  {SCORING_METRIC}: {mean_score:.4f} ± {std_score:.4f}")
    return cv_results


def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    tune,
    search_method,
    plot: bool = False,
):
    """Train and evaluate all available models."""
    all_models = {}
    all_metrics = {}
    for model_name in AVAILABLE_MODELS:
        model, metrics = train_model(
            model_name, X_train, y_train, X_test, y_test,
            tune=tune,
            search_method=search_method,
            plot=plot,
        )
        all_models[model_name] = model
        all_metrics[model_name] = metrics
    return all_models, all_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Credit Card Fraud Detection models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=AVAILABLE_MODELS,
        help="Model to train (default: random_forest)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train and compare all available models.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (GridSearchCV / RandomizedSearchCV).",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=TUNING_SEARCH_METHOD,
        choices=["grid", "random"],
        help=f"Tuning search strategy (default: {TUNING_SEARCH_METHOD}).",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run cross-validation after training.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot ROC curve and confusion matrix during inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(TRAINVAL_CSV, split=True)
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_val)}")

    X_train_resampled, y_train_resampled = resample_data(
        X_train, y_train, method=RESAMPLING_METHOD,
    )
    print(f"After resampling ({RESAMPLING_METHOD}): {len(X_train_resampled)} samples")

    X_train_scaled, X_val_scaled, scaler = scale_data(X_train_resampled, X_val)

    from inference import run_inference

    if args.all:
        all_models, all_metrics = train_all_models(
            X_train_scaled, y_train_resampled, X_val_scaled, y_val,
            tune=args.tune,
            search_method=args.search,
            plot=args.plot,
        )
        best_name = compare_models(all_metrics)

        best_model = all_models[best_name]
        save_model(best_model, os.path.join(MODEL_DIR, f"{best_name}.joblib"))

        metrics = run_inference(
            best_name, 
            scaler, 
            X_test, y_test, 
            plot=args.plot
        )
        print(f"\nFinal evaluation of best model ({best_name}) on test set:")
        print(metrics)
    else:
        model_name = args.model
        model, _metrics = train_model(
            model_name=model_name,
            X_train=X_train_scaled,
            y_train=y_train_resampled,
            X_test=X_val_scaled,
            y_test=y_val,
            tune=args.tune,
            search_method=args.search,
            plot=args.plot,
        )
        save_model(model, os.path.join(MODEL_DIR, f"{model_name}.joblib"))

        metrics = run_inference(
            model_name, 
            scaler, 
            X_test, y_test, 
            plot=args.plot
        )
        print(f"\nFinal evaluation of {model_name} on test set:")
        print(metrics)

    if args.cv:
        print("\n--- Cross-Validation ---")
        models_to_cv = AVAILABLE_MODELS if args.all else [args.model]
        for model_name in models_to_cv:
            cross_validate_model(model_name, X_train_scaled, y_train_resampled)

    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

