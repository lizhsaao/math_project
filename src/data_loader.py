import pandas as pd
import numpy as np
import sys
from src.config import DATA_PATH, TARGET, WIDTH

def inspect_data(df):
    """
    Performs a formal data integrity inspection. 
    Checks specific logical constraints for known datasets.
    """
    print(f"\n--- Data Integrity Inspection ---")
    
    # 1. LOGICAL CONSTRAINT CHECK (Exam Scores)
    if "Exam_Score" in df.columns:
        constraints = {
            "Exam_Score > 100": (df['Exam_Score'] > 100).sum(),
            "Attendance > 100%": (df['Attendance'] > 100).sum() if 'Attendance' in df.columns else 0,
            "Previous_Scores > 100": (df['Previous_Scores'] > 100).sum() if 'Previous_Scores' in df.columns else 0,
            "Sleep_Hours > 24": (df['Sleep_Hours'] > 24).sum() if 'Sleep_Hours' in df.columns else 0,
            "Negative Values": (df.select_dtypes(include=[np.number]) < 0).sum().sum()
        }
        for rule, count in constraints.items():
            status = "[FAIL]" if count > 0 else "[PASS]"
            print(f"  {status} {rule}: {count} issues found")
    
    # 2. CATEGORICAL SCHEMA DISCOVERY
    print(f"\n--- Categorical Schema (Manual Review) ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_vals = sorted(df[col].dropna().unique().astype(str))
        print(f"  {col}: {', '.join(unique_vals)}")
    
    # --- MANUAL GATEKEEPER ---
    user_input = input("Confirm categorical schema above (Y/N): ").strip().upper()
    if user_input != 'Y':
        print("\n[TERMINATED] Analysis halted for manual data review.")
        sys.exit()
    
    print("Schema confirmed. Proceeding...")

def load_and_clean():
    """
    Returns df_raw, df_cleaned, df_imputed, df_dropped
    """
    df_raw = pd.read_csv(DATA_PATH)
    
    # Remove rows where Target is missing (standard requirement)
    df_base = df_raw.dropna(subset=[TARGET]).copy()

    # Apply dataset-specific cleaning logic
    if TARGET == "Exam_Score":
        df_cleaned = df_base[df_base[TARGET] <= 100].copy()
    else:
        df_cleaned = df_base.copy()

    # Generate Tracks
    df_dropped = df_cleaned.dropna().copy()
    df_imputed = df_cleaned.copy()
    
    for col in df_imputed.columns[df_imputed.isnull().any()]:
        # Mode for categories, Median for numbers
        fill = df_imputed[col].mode()[0] if df_imputed[col].dtype == 'object' else df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(fill)
            
    return df_raw, df_cleaned, df_imputed, df_dropped