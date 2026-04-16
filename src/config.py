import os
from pathlib import Path

# root path
PROJECT_ROOT = Path(__file__).parent.parent


DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Data file paths
TRAIN_CSV = DATA_DIR / "train.csv"
TRAINVAL_CSV = DATA_DIR / "trainval.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# data cfg
TARGET_COL = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# resampling method
RESAMPLING_METHOD = "smote"  # "smote" or "random"

# cross-validation settings
CV_FOLDS = 5
SCORING_METRIC = "f1"

# hyperparameter tuning settings
TUNING_SEARCH_METHOD = "grid"   # "grid" or "random"
TUNING_N_ITER = 50              # iter for RandomizedSearchCV
TUNING_VERBOSE = 1              # 0=silent, 1=progress, 2=detailed

# Available models
AVAILABLE_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "adaboost",
    "svm",
    "knn",
    "voting",
]

