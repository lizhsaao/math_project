import pandas as pd
import numpy as np
import sys
from src.config import WIDTH

def inspect_data(df, target, limits):
    """
    Prints a data integrity report for manual review, then prompts the user
    to confirm the categorical schema before proceeding.

    Parameters
    ----------
    df     : Raw DataFrame (pre-cleaning).
    target : Name of the response variable column.
    limits : Dict mapping column names to (max_value, label) tuples.
    """
    print(f"\n--- Data Integrity Inspection ---")
    
    # 1. LOGICAL CONSTRAINT CHECK
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        # General Negative Value Check
        neg_count = (df[col] < 0).sum()
        status_neg = "[FAIL]" if neg_count > 0 else "[PASS]"
        print(f"  {status_neg} {col} (Negative Check): {neg_count} issues")

        # Upper Limit Check (Specific Columns from Config)
        if col in limits:
            limit_val, label = limits[col]
            upper_count = (df[col] > limit_val).sum()
            status_up = "[FAIL]" if upper_count > 0 else "[PASS]"
            print(f"  {status_up} {label}: {upper_count} issues")

    # 2. CATEGORICAL SCHEMA DISCOVERY
    print(f"\n--- Categorical Schema (Manual Review) ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_vals = sorted(df[col].dropna().unique().astype(str))
        print(f"  {col}: {', '.join(unique_vals)}")
    
    # --- MANUAL GATEKEEPER ---
    sys.__stdout__.write("\nConfirm categorical schema above (Y/N): ")
    sys.__stdout__.flush()
    user_input = sys.stdin.readline().strip().upper()
    if user_input != 'Y':
        print("\n[TERMINATED] Analysis halted for manual data review.")
        sys.exit()

    print("Schema confirmed. Proceeding...\n")

def load_and_clean(data_path, target, missing_cols, limits):
    """
    Loads the CSV, removes rows violating domain constraints, and generates
    the two analytical tracks.

    Parameters
    ----------
    data_path    : Path to the CSV file.
    target       : Name of the response variable column.
    missing_cols : List of columns used to define listwise deletion (Track B).
    limits       : Dict mapping column names to (max_value, label) tuples.

    Returns
    -------
    df_raw          : Original DataFrame as loaded.
    df_cleaned      : Post-constraint DataFrame (outliers removed).
    df_imputed_base : df_cleaned with NaNs retained (Track A base).
    df_dropped      : df_cleaned with missing rows removed (Track B).
    """
    df_raw = pd.read_csv(data_path)
    df_base = df_raw.dropna(subset=[target]).copy()

    # Clean based on the target's limit if it exists in config
    if target in limits:
        max_val = limits[target][0]
        df_cleaned = df_base[df_base[target] <= max_val].copy()
    else:
        df_cleaned = df_base.copy()

    df_dropped = df_cleaned.dropna(subset=missing_cols).copy() if missing_cols else df_cleaned.dropna().copy()
    df_imputed_base = df_cleaned.copy()
            
    return df_raw, df_cleaned, df_imputed_base, df_dropped