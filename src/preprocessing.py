"""
    Converts human-readable data into machine-readable matrices (X and y).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, TEST_SIZE


def preprocess_track(df, target, requires_imputation=False):
    """
    Splits, imputes, one-hot encodes, scales, and aligns one analytical track —
    all in the correct order to avoid leakage.

    Steps
    -----
    1. Split  — 80/20 train/test before any fitting or feature computation.
    2. Impute — Fill remaining NaNs using training-set statistics only
                (median for numeric, mode for categorical). The same fill
                values are applied to the test set.
    3. Encode — One-hot encode each split independently, then align the test
                matrix to the training columns (left join). Unseen test
                categories are dropped; missing train columns are filled with 0.
    4. Scale  — Standardise to zero mean / unit variance (fit on training only).

    Parameters
    ----------
    df                  : DataFrame for the track (Track A or B).
    target              : Name of the response variable column.
    requires_imputation : If True, fill NaNs from training statistics.
                          Set True for Track A (imputed), False for Track B
                          (listwise-deleted, so no NaNs remain).

    Returns
    -------
    X_train, X_test : Encoded feature matrices (80 / 20 split).
    y_train, y_test : Corresponding response vectors.
                      X_test is locked — use X_train for all CV and tuning.
    """
    # 1. Split fitst (Fixes Leakage)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 2. Separate features and target
    y_train = train_df[target]
    y_test  = test_df[target]
    X_train_raw = train_df.drop(columns=[target])
    X_test_raw  = test_df.drop(columns=[target])

    # 3. Imputation (training stats only, applied to both splits)
    if requires_imputation:
        for col in X_train_raw.columns:
            if (X_train_raw[col].dtype == 'object'
                    or X_train_raw[col].dtype.name == 'category'):
                fill_val = X_train_raw[col].mode()[0]
            else:
                fill_val = X_train_raw[col].median()
            X_train_raw[col] = X_train_raw[col].fillna(fill_val)
            X_test_raw[col]  = X_test_raw[col].fillna(fill_val)

    # 4. One-hot encoding + column alignment 
    X_train_enc = pd.get_dummies(X_train_raw, drop_first=True)
    X_test_enc  = pd.get_dummies(X_test_raw,  drop_first=True)

    # Align test to training columns (left join):
    #   - test category unseen in train → dropped
    #   - train column absent in test   → filled with 0
    X_train_aligned, X_test_aligned = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )

    # 5. Feature scaling (fit on training set only)
    # Standardising to zero mean / unit variance is essential for Lasso (which
    # penalises raw coefficient magnitudes) and beneficial for Linear Regression
    # (interpretable standardised coefficients). Tree-based models are
    # scale-invariant so this step does not affect DT or RF results.
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_aligned),
        columns=X_train_aligned.columns,
        index=X_train_aligned.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_aligned),
        columns=X_test_aligned.columns,
        index=X_test_aligned.index,
    )

    # X_test is locked — all CV and tuning use X_train only
    return X_train, X_test, y_train, y_test
