"""
    Transforms raw data into leak-free, machine-readable feature matrices and target vectors.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, TEST_SIZE, LOG_SKEWNESS_THRESHOLD


def preprocess_track(df, target, requires_imputation=False):
    """
    Splits, imputes, one-hot encodes, scales, and aligns one analytical track —
    all in the correct order to avoid leakage.

    Steps
    -----
    1. Split  — 80/20 train/test before any fitting or feature computation.
    2. Log    — If the training-set skewness of y exceeds LOG_SKEWNESS_THRESHOLD,
                apply log(y+1) to both splits. Only right-skewed targets are
                transformed (positive skewness > threshold). The caller receives
                a flag so it can back-transform predictions for reporting.
    3. Impute — Fill remaining NaNs using training-set statistics only
                (median for numeric, mode for categorical). The same fill
                values are applied to the test set.
    4. Encode — One-hot encode each split independently, then align the test
                matrix to the training columns (left join). Unseen test
                categories are dropped; missing train columns are filled with 0.
    5. Scale  — Standardise to zero mean / unit variance (fit on training only).

    Parameters
    ----------
    df : pandas.DataFrame
        The input data track.
    target : str
        Name of the target/response column.
    requires_imputation : bool, optional
        If True, fills NaNs using train-set median/mode (default False).

    Returns
    -------
    X_train, X_test : pandas.DataFrame
        Encoded and scaled feature matrices (80/20 split).
    y_train, y_test : pandas.Series
        Target vectors (log-transformed if highly skewed).
    log_target : bool
        True if a log(y+1) transform was applied to the target.
    prep_log : dict
        Logging metadata (skewness, imputed/encoded cols, dummy counts).
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

    # 2b. Apply log(y+1) if y_train skewness exceeds threshold (handles both tails).
    # Transformation is applied post-split to prevent data leakage.
    train_skewness = float(y_train.skew())
    log_target = abs(train_skewness) > LOG_SKEWNESS_THRESHOLD
    if log_target:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    # 3. Imputation (training stats only, applied to both splits)
    imputed_cols = (
        [col for col in X_train_raw.columns if X_train_raw[col].isna().any()]
        if requires_imputation else []
    )
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
    cat_cols_original = list(
        X_train_raw.select_dtypes(include=['object', 'category']).columns
    )
    X_train_enc = pd.get_dummies(X_train_raw, drop_first=True)
    X_test_enc  = pd.get_dummies(X_test_raw,  drop_first=True)
    # Net column change after encoding: total dummy cols minus original cat cols
    n_total_dummies = X_train_enc.shape[1] - (X_train_raw.shape[1] - len(cat_cols_original))

    # Align test to training columns (left join):
    #   - test category unseen in train -> dropped
    #   - train column absent in test -> filled with 0
    X_train_aligned, X_test_aligned = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )

    # 5. Feature scaling (fit on training set only)
    # Scale features (mean=0, var=1) on training set. Essential for Lasso/Linear Regression; 
    # optional for scale-invariant tree models (DT/RF).
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
    prep_log = {
        "skewness":        train_skewness,
        "log_target":      log_target,
        "imputed_cols":    imputed_cols,
        "encoded_cols":    cat_cols_original,
        "n_total_dummies": n_total_dummies,
    }
    return X_train, X_test, y_train, y_test, log_target, prep_log
