"""
    Entry point. Iterates over all datasets in DATASET_CONFIGS,
    runs the full pipeline, and saves a per-dataset report to results/.
"""
import sys
from src.config import DATA_DIR, RESULTS_DIR, DATASET_CONFIGS, WIDTH
from src.data_loader import load_and_clean, inspect_data
from src.preprocessing import preprocess_track

class Tee:
    """Writes to multiple streams at once — terminal and report file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for s in self.streams:
            s.write(text)
    def flush(self):
        for s in self.streams:
            s.flush()

def print_header(text, file=None):
    """
    Prints a centred section header.

    Parameters
    ----------
    text : Header text to display.
    file : Output stream; defaults to sys.stdout.
    """
    output = f"\n{'=' * WIDTH}\n{text.center(WIDTH, '=')}\n{'=' * WIDTH}\n"
    print(output, file=file or sys.stdout)

def main():
    """Runs the full pipeline for every dataset listed in DATASET_CONFIGS."""
    for filename, cfg in DATASET_CONFIGS.items():
        data_path = DATA_DIR / filename
        if not data_path.exists():
            print(f"Skipping {filename}: File not found.")
            continue

        # Create a text file to capture this dataset's output
        report_path = RESULTS_DIR / f"report_{filename.split('.')[0]}.txt"
        
        with open(report_path, "w") as f:
            print_header(f"Processing Dataset: {filename}", file=f)
            
            # 1. Load & Inspect
            df_raw, df_clean, df_imp, df_drop = load_and_clean(
                data_path, cfg['target'], cfg['missing_cols'], cfg['limits']
            )
            # Tee stdout -> terminal + report file so inspection output is visible
            # in the terminal (for the manual gatekeeper) AND saved to the report.
            old_stdout = sys.stdout
            sys.stdout = Tee(sys.__stdout__, f)
            inspect_data(df_raw, cfg['target'], cfg['limits'])
            sys.stdout = old_stdout

            # 2. Summary
            print(f"Cleaned Base: {df_clean.shape[0]} rows", file=f)
            print(f"Track A (Imputed): {df_imp.shape[0]} rows", file=f)
            print(f"Track B (Dropped): {df_drop.shape[0]} rows", file=f)

            # 3. Preprocessing
            print_header("3. Preprocessing & Encoding", file=f)
            X_ta, X_va, y_ta, y_va = preprocess_track(df_imp, requires_imputation=True, target=cfg['target'])
            X_tb, X_vb, y_tb, y_vb = preprocess_track(df_drop, requires_imputation=False, target=cfg['target'])
            
            print(f"Final Feature Count: {X_ta.shape[1]}", file=f)

        print(f"Done: {filename} (Report saved to {report_path})")

if __name__ == "__main__":
    main()