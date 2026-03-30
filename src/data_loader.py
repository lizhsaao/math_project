import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.config import WIDTH, SPARSITY_THRESHOLD

def inspect_data(df, target, limits):
    """
    Prints a data integrity report: negative/upper-limit checks for numeric
    columns and a categorical schema listing for manual review in the output log.

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
    _MAX_DISPLAY = 20   # show individual values only up to this many unique entries
    print(f"\n--- Categorical Schema (Manual Review) ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_vals = sorted(df[col].dropna().unique().astype(str))
        if len(unique_vals) <= _MAX_DISPLAY:
            print(f"  {col}: {', '.join(unique_vals)}")
        else:
            print(f"  {col}: [{len(unique_vals)} unique values — high cardinality, manual review recommended]")
    

def load_and_clean(data_path, target, limits):
    """
    Loads the CSV, normalises column names, removes rows violating domain
    constraints, and generates the two analytical tracks.

    Column names are normalised on load: spaces are replaced with underscores
    so that downstream code (config limits, preprocessing) can reference them
    consistently regardless of the raw CSV formatting.

    Missing columns for Track B (listwise deletion) are auto-detected from the
    cleaned dataframe — any column that contains at least one NaN is included.
    No manual specification in config is needed.

    Parameters
    ----------
    data_path : Path to the CSV file.
    target    : Name of the response variable column (post-normalisation).
    limits    : Dict mapping column names to (max_value, label) tuples.

    Returns
    -------
    df_raw          : DataFrame as loaded (column names already normalised).
    df_cleaned      : Post-constraint DataFrame (domain outliers removed).
    df_imputed_base : df_cleaned with NaNs retained — base for Track A.
    df_dropped      : df_cleaned with any NaN rows removed — base for Track B.
    """
    df_raw = pd.read_csv(data_path)

    # Normalise column names: spaces → underscores
    df_raw.columns = df_raw.columns.str.replace(' ', '_', regex=False)

    df_base = df_raw.dropna(subset=[target]).copy()

    # Clean based on the target's limit if it exists in config
    if target in limits:
        max_val = limits[target][0]
        df_cleaned = df_base[df_base[target] <= max_val].copy()
    else:
        df_cleaned = df_base.copy()

    # Drop columns that are too sparse to be useful predictors.
    # Columns above SPARSITY_THRESHOLD (e.g. monthly CPI in a daily dataset)
    # would destroy Track B by eliminating nearly all rows in the listwise step.
    predictor_cols = [c for c in df_cleaned.columns if c != target]
    sparse_cols = [
        c for c in predictor_cols
        if df_cleaned[c].isna().mean() > SPARSITY_THRESHOLD
    ]
    if sparse_cols:
        print(
            f"  [INFO] Dropped {len(sparse_cols)} sparse column(s) "
            f"(>{SPARSITY_THRESHOLD:.0%} missing): {sparse_cols}"
        )
        df_cleaned = df_cleaned.drop(columns=sparse_cols)

    # Auto-detect columns that still contain at least one missing value —
    # these define Track B (listwise deletion).
    missing_cols = [col for col in df_cleaned.columns if df_cleaned[col].isna().any()]

    df_dropped = df_cleaned.dropna(subset=missing_cols).copy() if missing_cols else df_cleaned.copy()
    df_imputed_base = df_cleaned.copy()

    return df_raw, df_cleaned, df_imputed_base, df_dropped


def calculate_vif(df, numeric_cols):
    """
    Calculates the Variance Inflation Factor (VIF) for each column in
    numeric_cols using statsmodels.stats.outliers_influence.

    VIF_i = 1 / (1 - R²_i), where R²_i is the R² from regressing predictor i
    on all other predictors in numeric_cols. Only complete cases are used.

    Categorical columns should be excluded before calling — pass only the
    numeric predictors you want to assess for multicollinearity.

    Parameters
    ----------
    df           : Cleaned DataFrame (post-outlier removal, pre-encoding).
    numeric_cols : List of numeric predictor column names to include.

    Returns
    -------
    pd.DataFrame with columns ["Feature", "VIF"] sorted by VIF descending.
    """
    if len(numeric_cols) < 2:
        print("  [INFO] VIF requires at least 2 numeric predictors — skipping.")
        return pd.DataFrame(columns=["Feature", "VIF"])

    X = df[numeric_cols].dropna().values
    n_obs = len(X)

    vif_values = [
        variance_inflation_factor(X, i) for i in range(len(numeric_cols))
    ]
    vif_df = (
        pd.DataFrame({"Feature": numeric_cols, "VIF": vif_values})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n--- VIF (Variance Inflation Factor, n={n_obs} complete cases) ---")
    print(f"  {'Feature':<38} {'VIF':>8}  Note")
    print(f"  {'-'*60}")
    for _, row in vif_df.iterrows():
        vif  = row["VIF"]
        note = "[SEVERE]   VIF >= 10" if vif >= 10 else ("[moderate] VIF >= 5" if vif >= 5 else "")
        vif_str = f"{vif:8.2f}" if np.isfinite(vif) else "     inf"
        print(f"  {row['Feature']:<38} {vif_str}  {note}")
    print(f"\n  [VIF > 5 = moderate collinearity;  VIF > 10 = severe]")

    return vif_df