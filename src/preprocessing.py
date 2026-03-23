"""
    Converts human-readable data into machine-readable matrices (X and y).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE

def preprocess_track(df, target, requires_imputation=False):
    """
    Splits, imputes (if required), one-hot encodes, and aligns one analytical track.

    Parameters
    ----------
    df                  : DataFrame for the track (Track A or B).
    target              : Name of the response variable column.
    requires_imputation : If True, imputes remaining NaNs using training-set
                          statistics only (mode for categorical, median for numeric).

    Returns
    -------
    X_train, X_test : Encoded feature matrices (80 / 20 split).
    y_train, y_test : Corresponding response vectors.
                      X_test is locked — use X_train for all CV and tuning.
    """
    # 1. SPLIT FIRST (Fixes Leakage)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 2. DYNAMIC IMPUTATION (On Training Data Only)
    if requires_imputation:
        for col in train_df.columns:
            if train_df[col].isnull().any():
                # Dynamically choose median or mode based on data type
                if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
                    fill_val = train_df[col].mode()[0]
                else:
                    fill_val = train_df[col].median()
                
                # Apply the train-derived value to BOTH sets
                train_df[col] = train_df[col].fillna(fill_val)
                test_df[col] = test_df[col].fillna(fill_val)

    # 3. DYNAMIC ONE-HOT ENCODING
    y_train = train_df[target]
    y_test = test_df[target]

    X_train_raw = train_df.drop(columns=[target])
    X_test_raw = test_df.drop(columns=[target])

    X_train_enc = pd.get_dummies(X_train_raw, drop_first=True)
    X_test_enc = pd.get_dummies(X_test_raw, drop_first=True)

    # Align test to training columns only (join="left").
    # Any category in test that wasn't in train is dropped.
    # Any train column missing from test is filled with 0.
    X_train_final, X_test_final = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )
    
    # X_test is locked — CV in main.py uses X_train only.
    return X_train_final, X_test_final, y_train, y_test