"""
    Entry point. Iterates over all datasets in DATASET_CONFIGS,
    runs the full pipeline, and saves a per-dataset report to results/.
"""
import sys
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

from src.config import DATA_DIR, RESULTS_DIR, DATASET_CONFIGS, WIDTH, DEPTHS, RANDOM_STATE
from src.data_loader import load_and_clean, inspect_data
from src.preprocessing import preprocess_track
from src.models import tune_decision_tree, get_metrics_and_preds
from src.visualiser import plot_tuning_curve, plot_actual_vs_predicted, plot_residuals, plot_feature_importance
from src.evaluator import create_summary_table

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

        # Define alphabetical prefix (e.g., results/StudentPerformanceFactors)
        prefix = RESULTS_DIR / filename.split('.')[0]
        report_path = f"{prefix}_report.txt"
        
        with open(report_path, "w") as f:
            print_header(f"Processing Dataset: {filename}", file=f)
            
            # 1. Load & Inspect
            print_header("1. Load & Inspect", file=f)
            df_raw, df_clean, df_imp, df_drop = load_and_clean(
                data_path, cfg['target'], cfg['missing_cols'], cfg['limits']
            )
            # Tee stdout -> terminal + report file so inspection output is visible
            old_stdout = sys.stdout
            sys.stdout = Tee(sys.__stdout__, f)
            inspect_data(df_raw, cfg['target'], cfg['limits'])
            sys.stdout = old_stdout

            # 2. Summary
            print_header("2. Summary", file=f)
            print(f"Cleaned Base: {df_clean.shape[0]} rows", file=f)
            print(f"Track A (Imputed): {df_imp.shape[0]} rows", file=f)
            print(f"Track B (Dropped): {df_drop.shape[0]} rows", file=f)

            # 3. Preprocessing (Standardised naming)
            print_header("3. Preprocessing & Encoding", file=f)
            X_ta, X_va, y_ta, y_va = preprocess_track(df_imp, requires_imputation=True, target=cfg['target'])
            X_tb, X_vb, y_tb, y_vb = preprocess_track(df_drop, requires_imputation=False, target=cfg['target'])

            def run_track(X_tr, X_te, y_tr, y_te, track_label):
                """Fits all 5 models; returns results, predictions, tuning history, and fitted optimal DT."""
                best_d, history = tune_decision_tree(X_tr, y_tr, DEPTHS)
                models_dict = {
                    "Null Model":         DummyRegressor(strategy="mean"),
                    "Linear Regression":  LinearRegression(),
                    "DT (Depth 3)":       DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
                    "DT (Unconstrained)": DecisionTreeRegressor(max_depth=None, random_state=RANDOM_STATE),
                    "DT (Optimal)":       DecisionTreeRegressor(max_depth=best_d, random_state=RANDOM_STATE),
                }
                results, preds = {}, {}
                for name, model in models_dict.items():
                    # Evaluate on 'locked' 20% test set
                    metrics, y_pred = get_metrics_and_preds(X_tr, y_tr, X_te, y_te, model)
                    results[name] = metrics
                    preds[name]   = y_pred
                    print(f"  {track_label}: Fitted {name}", file=f)
                # DT (Optimal) is already fitted inside get_metrics_and_preds
                return results, preds, history, models_dict["DT (Optimal)"]

            # Run both tracks before plotting so side-by-side figures can be produced
            results_a, preds_a, hist_a, opt_dt_a = run_track(X_ta, X_va, y_ta, y_va, "Track A")
            results_b, preds_b, hist_b, opt_dt_b = run_track(X_tb, X_vb, y_tb, y_vb, "Track B")

            print(f"Final Feature Count: {X_ta.shape[1]}", file=f)

            # 4. Modeling & Visualisation
            print_header("4. Model Evaluation & Visualisation", file=f)

            # LR hold-out RMSE used as the reference line in the tuning curve
            lr_rmse_a = results_a["Linear Regression"]["rmse"]
            lr_rmse_b = results_b["Linear Regression"]["rmse"]

            # Tuning curve — both tracks side by side
            plot_tuning_curve(hist_a, hist_b, f"{prefix}_DT_tuning.png", lr_rmse_a, lr_rmse_b)

            # Actual vs. predicted — one side-by-side figure per model
            for name in ["Null Model", "Linear Regression", "DT (Depth 3)", "DT (Unconstrained)", "DT (Optimal)"]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_actual_vs_predicted(
                    y_va, preds_a[name], y_vb, preds_b[name],
                    name, f"{prefix}_{clean}_actual_vs_pred.png"
                )

            # Residuals vs. predicted — LR and DT (Optimal) only
            for name in ["Linear Regression", "DT (Optimal)"]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_residuals(
                    y_va, preds_a[name], y_vb, preds_b[name],
                    name, f"{prefix}_{clean}_residuals.png"
                )

            # Feature importance — both tracks side by side, optimal DT only
            plot_feature_importance(
                opt_dt_a, opt_dt_b, X_ta.columns, X_tb.columns,
                f"{prefix}_feature_importance.png"
            )

            # 5. Final Comparison Table
            summary_df = create_summary_table(results_a, results_b)
            print("\nFinal Model Comparison (Strict 80/20 Test Evaluation):", file=f)
            print(summary_df.to_string(index=False), file=f)

        print(f"Done: {filename} (Report and graphs saved to {report_path})")

if __name__ == "__main__":
    main()