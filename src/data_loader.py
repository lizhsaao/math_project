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
    _MAX_DISPLAY = 20   # show individual values only up to this many
    
    print(f"\n--- Categorical Schema (Manual Review) ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        print("  No categorical columns detected.")
    
    for col in cat_cols:
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        if n_unique <= _MAX_DISPLAY:
            vals_str = ", ".join(sorted(str(v) for v in unique_vals))
            print(f"  {col}: {vals_str}")
        else:
            print(f"  {col}: [{n_unique} unique values — high cardinality, manual review recommended]")

    print("Schema confirmed. Proceeding...\n")

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
    load_meta       : Dict with row counts — n_raw, n_cols, n_target_dropped,
                      n_domain_removed, n_cleaned.
    """
    df_raw = pd.read_csv(data_path)

    # Normalise column names: spaces -> underscores
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

    load_meta = {
        "n_raw":            len(df_raw),
        "n_cols":           df_raw.shape[1],
        "n_target_dropped": len(df_raw) - len(df_base),        # rows with missing target
        "n_domain_removed": len(df_base) - len(df_cleaned),    # domain constraint violations
        "n_cleaned":        len(df_cleaned),
    }
    return df_raw, df_cleaned, df_imputed_base, df_dropped, load_meta


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

    # Add constant to prevent artificially high uncentered VIFs
    X_df = df[numeric_cols].dropna().copy()
    X_df.insert(0, 'const', 1.0)
    X = X_df.values
    n_obs = len(X)

    # Offset by 1 to skip the 'const' column at index 0
    vif_values = [
        variance_inflation_factor(X, i + 1) for i in range(len(numeric_cols))
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


def vif_drop_analysis(df, numeric_cols):
    """
    Demonstrates OLS estimator instability by identifying the single
    highest-VIF predictor, removing it, and re-computing VIF on the
    remaining predictors. The side-by-side comparison reveals how
    collinearity propagates through the (X'X)^(-1) matrix, inflating the
    variance of every OLS coefficient estimate in the model.

    Parameters
    ----------
    df           : Cleaned DataFrame (post-outlier removal, pre-encoding).
    numeric_cols : List of numeric predictor column names.

    Returns
    -------
    (top_feature, top_vif) : Name and VIF of the dropped predictor.
                             Both are None when < 3 numeric columns.
    """
    if len(numeric_cols) < 3:
        return None, None

    # Pin rows: use the same complete-case subset for BOTH computations so
    # the before/after VIF comparison is valid (different dropna() calls on
    # different column subsets could silently use different row counts).
    complete_mask = df[numeric_cols].notna().all(axis=1)
    base_df = df.loc[complete_mask]

    X_full_df = base_df[numeric_cols].copy()
    X_full_df.insert(0, 'const', 1.0)
    X_full = X_full_df.values

    # Compute full VIF for all predictors (offset +1 to skip 'const')
    vif_full = {
        col: variance_inflation_factor(X_full, i + 1)
        for i, col in enumerate(numeric_cols)
    }

    # Identify the highest-VIF predictor
    top_feature = max(vif_full, key=lambda c: (vif_full[c] if np.isfinite(vif_full[c]) else np.inf))
    top_vif     = vif_full[top_feature]

    # Re-compute VIF with that predictor removed — same rows as above
    reduced_cols = [c for c in numeric_cols if c != top_feature]
    X_reduced_df = base_df[reduced_cols].copy()
    X_reduced_df.insert(0, 'const', 1.0)
    X_reduced    = X_reduced_df.values

    vif_reduced  = {
        col: variance_inflation_factor(X_reduced, i + 1)
        for i, col in enumerate(reduced_cols)
    }

    # Print comparison table sorted by full-VIF descending
    sorted_cols = sorted(numeric_cols,
                         key=lambda c: vif_full[c] if np.isfinite(vif_full[c]) else np.inf,
                         reverse=True)

    top_vif_str = f"{top_vif:.2f}" if np.isfinite(top_vif) else "inf"
    print(f"\n--- VIF Drop Analysis: Demonstrating OLS Estimator Instability ---")
    print(f"  Highest-VIF predictor: '{top_feature}'  (VIF = {top_vif_str})")
    print(f"  Removing '{top_feature}' and re-computing VIF on remaining predictors:\n")
    print(f"  {'Feature':<38}  {'VIF (full)':>10}  {'VIF (reduced)':>14}  {'Δ VIF':>8}")
    print(f"  {'-'*76}")

    n_changed = 0
    for col in sorted_cols:
        v_full = vif_full[col]
        v_full_str = f"{v_full:10.2f}" if np.isfinite(v_full) else "       inf"
        if col == top_feature:
            print(f"  {col:<38}  {v_full_str}  {'[removed]':>14}  {'—':>8}")
        else:
            v_red = vif_reduced.get(col, float('nan'))
            v_red_str = f"{v_red:14.2f}" if np.isfinite(v_red) else "           inf"
            delta = v_red - v_full if np.isfinite(v_red) and np.isfinite(v_full) else float('nan')
            delta_str = f"{delta:+8.2f}" if np.isfinite(delta) else "     n/a"
            if np.isfinite(delta) and delta != 0:
                n_changed += 1
            print(f"  {col:<38}  {v_full_str}  {v_red_str}  {delta_str}")

    print(f"\n  [OLS Collinearity Effect]")
    print(f"  Removing a single predictor shifted the VIF of {n_changed}/{len(reduced_cols)}"
          f" remaining feature(s).")
    print(f"  High VIF inflates Var(β^) ∝ (X'X)^(-1): correlated predictors share explanatory")
    print(f"  power, making OLS coefficients sensitive to small perturbations in X.")
    print(f"  -> See Lasso Contrast in Section 4 for how L_1 regularisation resolves this.")

    return top_feature, top_vif