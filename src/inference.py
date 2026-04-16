"""
Inference script for Credit Card Fraud Detection.
Load a trained model and run predictions on unseen data (test split).
"""
import os


from config import MODEL_DIR
from data_helper import (
    plot_roc_curve, plot_confusion_matrix,
)
from utils import evaluate_model
from model import load_model, predict, predict_proba


# inference for test set 
def run_inference(model_name: str, scaler, X_test, y_test, plot: bool = False):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model at {model_path}. Run training.py first."
        )
    model = load_model(model_path)

    X_test_scaled = scaler.transform(X_test)

    y_pred = predict(model, X_test_scaled)
    y_scores = predict_proba(model, X_test_scaled)

    print(f"\nResults for model: {model_name}")
    print(f"Test samples: {len(y_test)}")
    metrics = evaluate_model(y_test, y_pred, y_scores)

    if plot:
        plot_roc_curve(y_test, y_scores)
        plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix")

    return metrics




