"""
    Central configuration for file paths, dataset schema rules, and machine learning hyperparameters.

    Constants
    ---------
    WIDTH : int
        Standard console width for printing headers.
    BASE_DIR, DATA_DIR, RESULTS_DIR : pathlib.Path
        Absolute paths for project root, input data, and output artifacts.
    DATASET_CONFIGS : dict
        Mapping of dataset filenames to their specific configurations:
            - target : str, name of the response variable.
            - bin_strategy : str, histogram binning method ('unit', 'integer', 'fd').
            - limits : dict, mapping of column names to upper-bound limits and error labels.
    SPARSITY_THRESHOLD : float
        Maximum acceptable missing value proportion; columns above this are dropped.
    LOG_SKEWNESS_THRESHOLD : float
        Absolute skewness limit; targets exceeding this receive a log(y+1) transform.
    RANDOM_STATE : int
        Global seed for reproducible splits and initializations.
    TEST_SIZE : float
        Proportion of data allocated to the hold-out test set (80/20 split).
    CV_FOLDS : int
        Number of cross-validation folds used for hyperparameter tuning.
    DEPTHS : list of int
        Candidate maximum tree depths for Decision Tree tuning.
    N_ESTIMATORS : list of int
        Candidate forest sizes for Random Forest tuning.
    LASSO_ALPHAS : list of float
        Log-spaced candidate regularization strengths for Lasso tuning.
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
        "bin_strategy": "unit",
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
        "bin_strategy": "fd",
        "limits": {
            "bedrooms": (10, "Bedrooms > 10"),
            "bathrooms": (10, "Bathrooms > 10"),
            "stories": (5, "Stories > 5"),
            "area": (20000, "Area > 20,000 sqft")
        }
    },
    "winequality-red.csv": {
        "target": "quality",
        "bin_strategy": "integer",
        "limits": {
            # Target: Score is 0-10
            "quality": (10, "Quality Score > 10"),
            
            # Predictor sanity checks based on chemical reality
            "pH": (14, "pH > 14 (Impossible)"),
            "alcohol": (20, "Alcohol > 20% (Likely Fortified/Error)"),
            "residual_sugar": (45, "Sugar > 45g/L (Sweet/Outlier for Red)"),
            "total_sulfur_dioxide": (300, "SO2 > 300ppm (Unlikely for table wine)")
        }
    }
}

# Drop sparse columns before Track A/B split to prevent low-coverage indicators from skewing Track B.
SPARSITY_THRESHOLD = 0.20 # 20 %

# Apply log(y+1) transform if target skewness exceeds threshold; 
# back-transform for all reported metrics and plots.
LOG_SKEWNESS_THRESHOLD = 0.75

# Global ML Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
DEPTHS = list(range(1, 21))
N_ESTIMATORS = list(range(10, 210, 20)) # 10, 30, ..., 190 — 10 candidates
LASSO_ALPHAS = list(np.logspace(-3, 2, 50))  # 0.001 -> 100 on log scale, 50 candidates