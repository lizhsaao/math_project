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
        "limits": {
            # Target Limit
            "Exam_Score": (100, "Score > 100"),
            # Predictor Limits
            "Attendance": (100, "Attendance > 100%"),
            "Sleep_Hours": (24, "Sleep > 24h/night"),
            "Previous_Scores": (100, "Previous Score > 100"),
            "Hours_Studied": (168, "Studying > 168h/week"),
            "Physical_Activity": (168, "Activity > 168h/week"),
            "Tutoring_Sessions": (60, "Tutoring > 60 sessions/month")
        }
    },
    "Housing.csv": {
        "target": "price",
        "limits": {
            "bedrooms": (10, "Bedrooms > 10"),
            "bathrooms": (10, "Bathrooms > 10"),
            "stories": (5, "Stories > 5"),
            "area": (20000, "Area > 20,000 sqft")
        }
    },
    "financial_regression.csv": {
        "target": "gold_close",
        "limits": {
            "gold_close": (5000, "Gold Price > 5000"),
            "oil_close": (200, "Oil Price > 200"),
            "silver_close": (100, "Silver Price > 100"),
            "sp500_close": (10000, "S&P 500 > 10,000"),
            "us_rates_%": (20, "Interest Rates > 20%")
        }
    }
}

# Global ML Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
DEPTHS = list(range(1, 21))
N_ESTIMATORS = list(range(10, 310, 10)) # 10, 20, ..., 300 for RF tuning curve
LASSO_ALPHAS = list(np.logspace(-3, 2, 100)) # 0.001 -> 100 on log scale, 100 candidates