"""
    Central configuration: file paths, dataset-specific cleaning rules,
    and global ML hyperparameters.
"""
import numpy as np
from pathlib import Path

WIDTH = 50
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset-specific rules
DATASET_CONFIGS = {
    "StudentPerformanceFactors.csv": {
        "target": "Exam_Score",
        "missing_cols": ["Parental_Education_Level", "Teacher_Quality", "Distance_from_Home"],
        "limits": {
            "Exam_Score": (100, "Score > 100"),
            "Attendance": (100, "Attendance > 100%"),
            "Sleep_Hours": (24, "Sleep > 24h")
        }
    },
    # Add more here later
}

# Global ML Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
DEPTHS = list(range(1, 21))
N_ESTIMATORS = list(range(10, 310, 10)) # 10, 20, ..., 300 for RF tuning curve
LASSO_ALPHAS = list(np.logspace(-3, 2, 100)) # 0.001 -> 100 on log scale, 100 candidates