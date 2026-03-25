"""
    Entry point. Iterates over all datasets in DATASET_CONFIGS,
    runs the full pipeline, and saves all outputs (report + plots)
    into a per-dataset subdirectory under results/.
"""
import sys
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from src.config import DATA_DIR, RESULTS_DIR, DATASET_CONFIGS, WIDTH, DEPTHS, N_ESTIMATORS, RANDOM_STATE
from src.data_loader import load_and_clean, inspect_data
from src.preprocessing import preprocess_track
from src.models import tune_decision_tree, tune_random_forest, get_metrics_and_preds, cv_rmse_lr
from src.visualiser import (plot_exam_score_distribution, plot_correlation_with_target,
                            plot_tuning_curve, plot_actual_vs_predicted,
                            plot_residuals, plot_feature_importance)
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

        # One subdirectory per dataset (e.g., results/StudentPerformanceFactors/)
        out_dir = RESULTS_DIR / filename.split('.')[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.txt"

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

            # EDA plots
            plot_exam_score_distribution(df_clean, cfg['target'], out_dir / "score_distribution.png")
            plot_correlation_with_target(df_imp, df_drop, cfg['target'], out_dir / "correlation.png")

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
                """Fits all 6 models; returns results, predictions, DT history, RF history,
                fitted optimal DT, and fitted optimal RF."""
                best_d, dt_history = tune_decision_tree(X_tr, y_tr, DEPTHS)
                best_n, rf_history = tune_random_forest(X_tr, y_tr, N_ESTIMATORS)
                models_dict = {
                    "Null Model":         DummyRegressor(strategy="mean"),
                    "Linear Regression":  LinearRegression(),
                    "DT (Depth 3)":       DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
                    "DT (Unconstrained)": DecisionTreeRegressor(max_depth=None, random_state=RANDOM_STATE),
                    "DT (Optimal)":       DecisionTreeRegressor(max_depth=best_d, random_state=RANDOM_STATE),
                    "Random Forest":      RandomForestRegressor(n_estimators=best_n, random_state=RANDOM_STATE, n_jobs=-1),
                }
                results, preds = {}, {}
                for name, model in models_dict.items():
                    # Evaluate on 'locked' 20% test set
                    metrics, y_pred = get_metrics_and_preds(X_tr, y_tr, X_te, y_te, model)
                    results[name] = metrics
                    preds[name]   = y_pred
                    print(f"  {track_label}: Fitted {name}", file=f)
                print("\n", file=f)
                # Models are fitted in-place by get_metrics_and_preds
                return results, preds, dt_history, rf_history, models_dict["DT (Optimal)"], models_dict["Random Forest"]

            # Run both tracks before plotting so side-by-side figures can be produced
            results_a, preds_a, hist_dt_a, hist_rf_a, opt_dt_a, opt_rf_a = run_track(X_ta, X_va, y_ta, y_va, "Track A")
            results_b, preds_b, hist_dt_b, hist_rf_b, opt_dt_b, opt_rf_b = run_track(X_tb, X_vb, y_tb, y_vb, "Track B")

            print(f"Final Feature Count: {X_ta.shape[1]}", file=f)

            # Optimal DT depths — plain text to report, ANSI colour to terminal
            depth_a = opt_dt_a.max_depth
            depth_b = opt_dt_b.max_depth
            print(f"Optimal DT depth — Track A: {depth_a},  Track B: {depth_b}", file=f)
            sys.__stdout__.write(
                f"  Optimal DT depth — \033[34mTrack A: {depth_a}\033[0m,"
                f"  \033[32mTrack B: {depth_b}\033[0m\n"
            )

            # 4. Modeling & Visualisation
            print_header("4. Model Evaluation & Visualisation", file=f)

            # LR CV RMSE on training set — same metric as the DT curve, so the
            # baseline sits on a comparable scale
            lr_rmse_a = cv_rmse_lr(X_ta, y_ta)
            lr_rmse_b = cv_rmse_lr(X_tb, y_tb)

            # DT tuning curve — CV RMSE vs tree depth
            plot_tuning_curve(hist_dt_a, hist_dt_b, out_dir / "DT_tuning.png", lr_rmse_a, lr_rmse_b)

            # RF tuning curve — CV RMSE vs n_estimators
            plot_tuning_curve(
                hist_rf_a, hist_rf_b, out_dir / "RF_tuning.png", lr_rmse_a, lr_rmse_b,
                x_key="n_estimators", x_label="Number of Trees",
                suptitle="CV RMSE vs. Number of Trees  (10-fold CV)"
            )

            # Actual vs. predicted — one side-by-side figure per model
            for name in ["Null Model", "Linear Regression", "DT (Depth 3)", "DT (Unconstrained)", "DT (Optimal)", "Random Forest"]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_actual_vs_predicted(
                    y_va, preds_a[name], y_vb, preds_b[name],
                    name, out_dir / f"{clean}_actual_vs_pred.png"
                )

            # Residuals vs. predicted — LR, DT (Optimal), and RF
            for name in ["Linear Regression", "DT (Optimal)", "Random Forest"]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_residuals(
                    y_va, preds_a[name], y_vb, preds_b[name],
                    name, out_dir / f"{clean}_residuals.png"
                )

            # Feature importance — DT (Optimal)
            plot_feature_importance(
                opt_dt_a, opt_dt_b, X_ta.columns, X_tb.columns,
                out_dir / "DT_feature_importance.png"
            )

            # Feature importance — Random Forest (more stable; averaged across all trees)
            plot_feature_importance(
                opt_rf_a, opt_rf_b, X_ta.columns, X_tb.columns,
                out_dir / "RF_feature_importance.png",
                model_name="Random Forest"
            )

            # 5. Final Comparison Table
            summary_df = create_summary_table(results_a, results_b)
            print("\nFinal Model Comparison (Strict 80/20 Test Evaluation):", file=f)
            print(summary_df.to_string(index=False), file=f)

        print(f"Done: {filename} → {out_dir}/")

if __name__ == "__main__":
    main()