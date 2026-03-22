from src.config import WIDTH
from src.data_loader import load_and_clean, inspect_data

def print_header(text):
    """Prints a perfectly centered and aligned section header using the global WIDTH."""
    print("\n" + "=" * WIDTH)
    print(f" {text} ".center(WIDTH, "="))
    print("=" * WIDTH)

def print_divider():
    """Prints a simple dashed line for sub-sections."""
    print("-" * WIDTH)

def main():
    # --- SECTION 1: DATA LOADING & INSPECTION ---
    print_header("1. Data Loading & Inspection")
    
    df_raw, df_cleaned, df_imputed, df_dropped = load_and_clean()
    
    # Run the "Eye-Test" and Logic Check from data_loader.py
    inspect_data(df_raw)

    # --- SECTION 2: EXPERIMENTAL TRACK SUMMARY ---
    print_header("2. Experimental Track Summary")
    
    print(f"Original Dataset:  {df_raw.shape[0]} rows")
    print(f"Cleaned Base:      {df_cleaned.shape[0]} rows (post-outlier)")
    print_divider()
    
    rows_lost = df_cleaned.shape[0] - df_dropped.shape[0]
    loss_pct = rows_lost / df_cleaned.shape[0] * 100

    print(f"Track A (Imputed): {df_imputed.shape[0]} rows")
    print(f"Track B (Dropped): {df_dropped.shape[0]} rows")
    print(f"Track B Loss:      {rows_lost} rows removed from cleaned data ({loss_pct:.2f}%)")

    # --- SECTION 3: PREPROCESSING ---
    print_header("3. Preprocessing & Encoding")
    # This is our next step: 
    # X_a, y_a, X_b, y_b = preprocess_data(df_imputed, df_dropped)

if __name__ == "__main__":
    main()