import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import MISSING_COLS


def load_and_clean(path):
    """Load CSV and remove rows with impossible Exam_Score values (> 100)."""
    df = pd.read_csv(path)
    df = df[df["Exam_Score"] <= 100].copy()
    return df


def make_track_b(df):
    """Listwise deletion: drop rows with any missing value in MISSING_COLS."""
    return df.dropna(subset=MISSING_COLS).copy()


def split_and_encode(df, target, test_size, random_state):
    """
    80/20 train-test split -> mode imputation on train only (for any remaining
    NaNs) -> one-hot encoding with drop_first=True -> column alignment.

    Returns X_train, X_test, y_train, y_test (all encoded).
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Mode imputation
    for col in X_train.columns[X_train.isna().any()]:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    # One-hot encoding
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, drop_first=True)

    # Align test columns to train (fills gaps with 0)
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )

    return X_train_enc, X_test_enc, y_train, y_test
