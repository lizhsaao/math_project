from pathlib import Path

WIDTH = 50

DATA_PATH = Path(__file__).parent.parent / "data" / "StudentPerformanceFactors.csv"

TARGET = "Exam_Score"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

MISSING_COLS = [
    "Parental_Education_Level",
    "Teacher_Quality",
    "Distance_from_Home",
]

DEPTHS = list(range(1, 21))
