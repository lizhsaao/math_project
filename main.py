"""
    Entry point. Iterates over all datasets in DATASET_CONFIGS,
    runs the full pipeline, and saves all outputs (report + plots)
    into a per-dataset subdirectory under results/.
"""
import sys
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from src.config import (DATA_DIR, RESULTS_DIR, DATASET_CONFIGS, WIDTH,
                        DEPTHS, N_ESTIMATORS, LASSO_ALPHAS, RANDOM_STATE)
from src.data_loader import load_and_clean, inspect_data
from src.preprocessing import preprocess_track
from src.models import tune_decision_tree, tune_random_forest, tune_lasso, get_metrics_and_preds, cv_rmse_lr
from src.visualiser import (plot_exam_score_distribution, plot_correlation_with_target,
                            plot_tuning_curve, plot_actual_vs_predicted,
                            plot_residuals, plot_feature_importance,
                            plot_lasso_tuning_curve, plot_lasso_coefficients)
from src.evaluator import create_summary_table

# Canonical model order used as the default when models=None is passed to main().
ALL_MODELS = [
    "Null Model", "Linear Regression", "Lasso",
    "DT (Depth 3)", "DT (Unconstrained)", "DT (Optimal)", "Random Forest",
]

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

def main(models=None, datasets=None):
    """
    Runs the modelling pipeline for datasets listed in DATASET_CONFIGS.

    Parameters
    ----------
    models   : Which models to fit and evaluate.
               - None (default): all models in ALL_MODELS.
               - str: a single model name, e.g. "Random Forest".
               - list[str]: an explicit ordered subset, e.g. ["Linear Regression", "Lasso"].
               Only the requested models are tuned, fitted, and plotted, so omitting
               slow models (e.g. "Random Forest") speeds up the run significantly.
    datasets : Which datasets (filenames) to process.
               - None (default): all entries in DATASET_CONFIGS.
               - str: a single filename, e.g. "Housing.csv".
               - list[str]: an explicit subset, e.g. ["Housing.csv"].
    """
    # Resolve models parameter to an ordered list
    if models is None:
        model_list = ALL_MODELS
    elif isinstance(models, str):
        model_list = [models]
    else:
        model_list = list(models)

    # Resolve datasets parameter to a set for O(1) lookup
    if datasets is None:
        dataset_set = set(DATASET_CONFIGS.keys())
    elif isinstance(datasets, str):
        dataset_set = {datasets}
    else:
        dataset_set = set(datasets)

    for filename, cfg in DATASET_CONFIGS.items():
        if filename not in dataset_set:
            continue
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
                data_path, cfg['target'], cfg['limits']
            )

            # Detect single-track mode early so EDA plots can use it.
            # When no rows were dropped (no missing values), both tracks are
            # identical — run only one.
            single_track = df_imp.shape[0] == df_drop.shape[0]

            # Tee stdout -> terminal + report file so inspection output is visible
            old_stdout = sys.stdout
            sys.stdout = Tee(sys.__stdout__, f)
            inspect_data(df_raw, cfg['target'], cfg['limits'])
            sys.stdout = old_stdout

            # EDA plots
            plot_exam_score_distribution(df_clean, cfg['target'], out_dir / "score_distribution.png")
            plot_correlation_with_target(df_imp, df_drop, cfg['target'], out_dir / "correlation.png",
                                         single_track=single_track)

            # 2. Summary
            print_header("2. Summary", file=f)
            print(f"Cleaned Base: {df_clean.shape[0]} rows", file=f)
            if single_track:
                print(f"Rows: {df_imp.shape[0]} (no missing values — single-track mode)", file=f)
            else:
                print(f"Track A (Imputed): {df_imp.shape[0]} rows", file=f)
                print(f"Track B (Dropped): {df_drop.shape[0]} rows", file=f)

            # 3. Preprocessing & Encoding
            print_header("3. Preprocessing & Encoding", file=f)
            X_ta, X_va, y_ta, y_va = preprocess_track(df_imp,  requires_imputation=True,  target=cfg['target'])
            if not single_track:
                X_tb, X_vb, y_tb, y_vb = preprocess_track(df_drop, requires_imputation=False, target=cfg['target'])

            def run_track(X_tr, X_te, y_tr, y_te, track_label):
                """
                Tunes and fits only the models in model_list, then evaluates each
                on the held-out 20% test set.

                Returns
                -------
                results  : dict  model_name -> {"rmse", "r2"}
                preds    : dict  model_name -> y_pred array
                histories: dict  tuning key -> history list
                           Keys: "DT" | "RF" | "Lasso" (only present if tuning ran)
                fitted   : dict  model_name -> fitted estimator
                           Only models with tunable hyperparameters are stored here.
                """
                # --- Conditional tuning (only for requested models) ---
                histories = {}

                best_d = None
                if "DT (Optimal)" in model_list:
                    best_d, dt_hist = tune_decision_tree(X_tr, y_tr, DEPTHS)
                    histories["DT"] = dt_hist

                best_n = None
                if "Random Forest" in model_list:
                    best_n, rf_hist = tune_random_forest(X_tr, y_tr, N_ESTIMATORS)
                    histories["RF"] = rf_hist

                best_alpha = None
                if "Lasso" in model_list:
                    best_alpha, lasso_hist = tune_lasso(X_tr, y_tr, LASSO_ALPHAS)
                    histories["Lasso"] = lasso_hist

                # --- Full catalogue; filter to requested models only ---
                catalogue = {
                    "Null Model":         DummyRegressor(strategy="mean"),
                    "Linear Regression":  LinearRegression(),
                    "Lasso":              Lasso(alpha=best_alpha, max_iter=10_000) if best_alpha is not None else None,
                    "DT (Depth 3)":       DecisionTreeRegressor(max_depth=3,    random_state=RANDOM_STATE),
                    "DT (Unconstrained)": DecisionTreeRegressor(max_depth=None, random_state=RANDOM_STATE),
                    "DT (Optimal)":       DecisionTreeRegressor(max_depth=best_d, random_state=RANDOM_STATE) if best_d is not None else None,
                    "Random Forest":      RandomForestRegressor(n_estimators=best_n, random_state=RANDOM_STATE, n_jobs=-1) if best_n is not None else None,
                }
                models_to_run = {k: v for k, v in catalogue.items()
                                 if k in model_list and v is not None}

                results, preds, fitted = {}, {}, {}
                for name, model in models_to_run.items():
                    metrics, y_pred = get_metrics_and_preds(X_tr, y_tr, X_te, y_te, model)
                    results[name] = metrics
                    preds[name]   = y_pred
                    fitted[name]  = model
                    prefix = f"  {track_label}: " if track_label else "  "
                    print(f"{prefix}Fitted {name}", file=f)
                print("\n", file=f)
                return results, preds, histories, fitted

            # Run tracks — skip Track B entirely when data are identical
            label_a = "" if single_track else "Track A"
            results_a, preds_a, hist_a, fitted_a = run_track(X_ta, X_va, y_ta, y_va, label_a)
            if single_track:
                results_b, preds_b, hist_b, fitted_b = results_a, preds_a, hist_a, fitted_a
                X_tb, X_vb, y_tb, y_vb = X_ta, X_va, y_ta, y_va
            else:
                results_b, preds_b, hist_b, fitted_b = run_track(X_tb, X_vb, y_tb, y_vb, "Track B")

            print(f"Final Feature Count: {X_ta.shape[1]}", file=f)

            # Optimal hyperparameters — plain text to report, ANSI colour to terminal
            if "DT (Optimal)" in fitted_a:
                depth_a = fitted_a["DT (Optimal)"].max_depth
                if single_track:
                    print(f"Optimal DT depth — {depth_a}", file=f)
                    sys.__stdout__.write(f"  Optimal DT depth — \033[34m{depth_a}\033[0m\n")
                else:
                    depth_b = fitted_b["DT (Optimal)"].max_depth
                    print(f"Optimal DT depth — Track A: {depth_a},  Track B: {depth_b}", file=f)
                    sys.__stdout__.write(
                        f"  Optimal DT depth — \033[34mTrack A: {depth_a}\033[0m,"
                        f"  \033[32mTrack B: {depth_b}\033[0m\n"
                    )

            if "Lasso" in fitted_a:
                alpha_a = fitted_a["Lasso"].alpha
                if single_track:
                    print(f"Optimal Lasso α  — {alpha_a:.4g}", file=f)
                    sys.__stdout__.write(f"  Optimal Lasso α  — \033[34m{alpha_a:.4g}\033[0m\n")
                else:
                    alpha_b = fitted_b["Lasso"].alpha
                    print(f"Optimal Lasso α  — Track A: {alpha_a:.4g},  Track B: {alpha_b:.4g}", file=f)
                    sys.__stdout__.write(
                        f"  Optimal Lasso α  — \033[34mTrack A: {alpha_a:.4g}\033[0m,"
                        f"  \033[32mTrack B: {alpha_b:.4g}\033[0m\n"
                    )

            if "Random Forest" in fitted_a:
                n_a = fitted_a["Random Forest"].n_estimators
                if single_track:
                    print(f"Optimal n_estimators — {n_a}", file=f)
                    sys.__stdout__.write(f"  Optimal n_estimators — \033[34m{n_a}\033[0m\n")
                else:
                    n_b = fitted_b["Random Forest"].n_estimators
                    print(f"Optimal n_estimators — Track A: {n_a},  Track B: {n_b}", file=f)
                    sys.__stdout__.write(
                        f"  Optimal n_estimators — \033[34mTrack A: {n_a}\033[0m,"
                        f"  \033[32mTrack B: {n_b}\033[0m\n"
                    )

            # 4. Modeling & Visualisation
            print_header("4. Model Evaluation & Visualisation", file=f)

            # LR baseline — needed for DT, RF, and Lasso tuning curves
            needs_lr_baseline = any(k in hist_a for k in ("DT", "RF", "Lasso"))
            lr_rmse_a = cv_rmse_lr(X_ta, y_ta) if needs_lr_baseline else None
            lr_rmse_b = cv_rmse_lr(X_tb, y_tb) if (needs_lr_baseline and not single_track) else None

            # Tuning curves — only drawn when the corresponding tuning was performed
            if "DT" in hist_a:
                plot_tuning_curve(hist_a["DT"], hist_b["DT"],
                                  out_dir / "DT_tuning.png", lr_rmse_a, lr_rmse_b,
                                  single_track=single_track)

            if "RF" in hist_a:
                plot_tuning_curve(hist_a["RF"], hist_b["RF"],
                                  out_dir / "RF_tuning.png", lr_rmse_a, lr_rmse_b,
                                  x_key="n_estimators", x_label="Number of Trees",
                                  suptitle="CV RMSE vs. Number of Trees  (10-fold CV)",
                                  single_track=single_track)

            if "Lasso" in hist_a:
                plot_lasso_tuning_curve(hist_a["Lasso"], hist_b["Lasso"],
                                        out_dir / "Lasso_tuning.png",
                                        lr_rmse_a=lr_rmse_a, lr_rmse_b=lr_rmse_b,
                                        single_track=single_track)

            # Actual vs. predicted — one figure per fitted model
            for name in [m for m in model_list if m in preds_a]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_actual_vs_predicted(
                    y_va, preds_a[name], y_vb, preds_b[name],
                    name, out_dir / f"{clean}_actual_vs_pred.png",
                    single_track=single_track, target=cfg['target']
                )

            # Residuals — drawn for fitted models among the standard candidates
            for name in [m for m in ["Linear Regression", "Lasso", "DT (Optimal)", "Random Forest"]
                         if m in preds_a]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_residuals(
                    y_va, preds_a[name], y_vb, preds_b[name],
                    name, out_dir / f"{clean}_residuals.png",
                    single_track=single_track, target=cfg['target']
                )

            # Feature importances — only when the relevant model was fitted
            if "DT (Optimal)" in fitted_a:
                plot_feature_importance(
                    fitted_a["DT (Optimal)"], fitted_b["DT (Optimal)"],
                    X_ta.columns, X_tb.columns,
                    out_dir / "DT_feature_importance.png",
                    single_track=single_track
                )

            if "Lasso" in fitted_a:
                plot_lasso_coefficients(
                    fitted_a["Lasso"], fitted_b["Lasso"],
                    X_ta.columns, X_tb.columns,
                    out_dir / "Lasso_coefficients.png",
                    single_track=single_track
                )

            if "Random Forest" in fitted_a:
                plot_feature_importance(
                    fitted_a["Random Forest"], fitted_b["Random Forest"],
                    X_ta.columns, X_tb.columns,
                    out_dir / "RF_feature_importance.png",
                    model_name="Random Forest", single_track=single_track
                )

            # 5. Final Comparison Table
            summary_df = create_summary_table(results_a, results_b, single_track=single_track)
            print("\nFinal Model Comparison (Strict 80/20 Test Evaluation):", file=f)
            print(summary_df.to_string(index=False), file=f)

        print(f"Done: {filename} -> {out_dir}/")

if __name__ == "__main__":
    # main(datasets = "winequality-red.csv")
    main()