"""
    Entry point. Iterates over all datasets in DATASET_CONFIGS,
    runs the full pipeline, and saves all outputs (report + plots)
    into a per-dataset subdirectory under results/.
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

from src.config import (DATA_DIR, RESULTS_DIR, DATASET_CONFIGS, WIDTH,
                        DEPTHS, N_ESTIMATORS, LASSO_ALPHAS, RANDOM_STATE,
                        LOG_SKEWNESS_THRESHOLD)
from src.data_loader import load_and_clean, inspect_data, calculate_vif, vif_drop_analysis
from src.preprocessing import preprocess_track
from src.models import tune_decision_tree, tune_random_forest, tune_lasso, get_metrics_and_preds, cv_rmse_lr
from src.visualiser import (plot_exam_score_distribution, plot_correlation_with_target,
                            plot_tuning_curve, plot_actual_vs_predicted,
                            plot_residuals, plot_feature_importance,
                            plot_lasso_tuning_curve, plot_lasso_coefficients,
                            plot_outlier_profile, plot_residual_normality)
from src.evaluator import (create_summary_table, residual_normality_descriptives,
                            residual_outlier_profile)

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

def _print_lasso_contrast(fitted_a, fitted_b, X_ta, X_tb,
                           vif_top_feature, vif_top_vif, single_track, f, tag=""):
    """
    Prints a Lasso L_1 vs OLS collinearity narrative to the report file.

    Shows the non-zero coefficient table for each track, cross-references the
    highest-VIF feature from the VIF drop analysis, and explains how the L_1
    penalty resolves the (X'X)^(-1) variance inflation that destabilises OLS.
    """
    import pandas as pd

    def _coef_table(model, feat_names, label):
        coefs   = pd.Series(model.coef_, index=feat_names)
        nonzero = coefs[coefs != 0].sort_values(key=abs, ascending=False)
        zero    = coefs[coefs == 0]
        print(f"\n  {label}: {len(nonzero)}/{len(coefs)} features retained  "
              f"(λ = {model.alpha:.5g})", file=f)
        print(f"  {'Feature':<38}  {'β (Lasso)':>12}", file=f)
        print(f"  {'-'*55}", file=f)
        for feat, coef in nonzero.items():
            print(f"  {feat:<38}  {coef:+12.4f}", file=f)
        if len(zero):
            print(f"  Zeroed-out ({len(zero)}):  {', '.join(zero.index.tolist())}", file=f)
        return coefs

    print(f"\n--- Lasso Contrast: L_1 Regularisation as Multicollinearity Resolution ---",
          file=f)
    print(f"  Objective: minimise  RSS(β) + λ·Σ|β_j|", file=f)
    print(f"  The L_1 penalty shrinks redundant coefficients to exactly zero,", file=f)
    print(f"  performing automated feature selection where OLS estimator variance explodes.", file=f)

    coefs_a = _coef_table(fitted_a["Lasso"], X_ta.columns, "Track A")
    if not single_track:
        _coef_table(fitted_b["Lasso"], X_tb.columns, "Track B")

    if vif_top_feature and vif_top_feature in coefs_a.index:
        c = coefs_a[vif_top_feature]
        verdict = (f"β = {c:+.4f}  (retained)"
                   if c != 0 else "β = 0.0000  (eliminated by L_1 penalty)")
        vif_str = f"{vif_top_vif:.2f}" if vif_top_vif and np.isfinite(vif_top_vif) else "inf"
        print(f"\n  VIF cross-reference: '{vif_top_feature}' "
              f"(highest VIF = {vif_str}) -> {verdict}", file=f)

    print(f"\n  The L_1 constraint forces the optimiser to select at most one", file=f)
    print(f"  predictor from a correlated group, resolving the (X'X)^(-1) variance", file=f)
    print(f"  inflation that destabilises OLS where multicollinearity is present.", file=f)
    suffix = f"_{tag}" if tag else ""
    print(f"  -> See Lasso_coefficients{suffix}.png for the full sparse coefficient profile.", file=f)


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
        tag = filename.split('.')[0]   # e.g. "StudentPerformanceFactors", "Housing", "winequality-red"
        report_path = out_dir / f"report_{tag}.txt"

        with open(report_path, "w") as f:
            print_header(f"Processing Dataset: {filename}", file=f)
            sys.__stdout__.write(f"\n{filename}\n")

            # 1. Load & Inspect
            print_header("1. Load & Inspect", file=f)
            sys.__stdout__.write("  1. Load & Inspect\n")
            df_raw, df_clean, df_imp, df_drop, load_meta = load_and_clean(
                data_path, cfg['target'], cfg['limits']
            )

            # Dataset shape and row-drop summary (printed directly to report)
            print(f"  Dataset:    {load_meta['n_raw']} rows × {load_meta['n_cols']} columns (raw)", file=f)
            if load_meta['n_target_dropped'] > 0:
                print(f"  Dropped:    {load_meta['n_target_dropped']} row(s) — missing target value", file=f)
            if load_meta['n_domain_removed'] > 0:
                print(f"  Dropped:    {load_meta['n_domain_removed']} row(s) — domain constraint violations", file=f)
            print(f"  Clean base: {load_meta['n_cleaned']} rows\n", file=f)

            # Detect single-track mode early so EDA plots can use it.
            # When no rows were dropped (no missing values), both tracks are
            # identical — run only one.
            single_track = df_imp.shape[0] == df_drop.shape[0]

            # Redirect stdout -> report file only (terminal shows progress lines)
            old_stdout = sys.stdout
            sys.stdout = f
            inspect_data(df_raw, cfg['target'], cfg['limits'])
            numeric_pred_cols = [
                c for c in df_clean.select_dtypes(include=[np.number]).columns
                if c != cfg['target']
            ]
            calculate_vif(df_clean, numeric_pred_cols)
            vif_top_feature, vif_top_vif = vif_drop_analysis(df_clean, numeric_pred_cols)
            sys.stdout = old_stdout

            # EDA plots
            plot_exam_score_distribution(df_clean, cfg['target'], out_dir / f"score_distribution_{tag}.png",
                                         bin_strategy=cfg.get('bin_strategy', 'fd'))
            print(f"  [Plot saved]  score_distribution_{tag}.png", file=f)
            plot_correlation_with_target(df_imp, df_drop, cfg['target'], out_dir / f"correlation_{tag}.png",
                                         single_track=single_track)
            print(f"  [Plot saved]  correlation_{tag}.png", file=f)

            # 2. Summary
            print_header("2. Summary", file=f)
            sys.__stdout__.write("  2. Summary\n")
            print(f"Cleaned Base: {df_clean.shape[0]} rows", file=f)
            if single_track:
                print(f"Rows: {df_imp.shape[0]} (no missing values — single-track mode)", file=f)
            else:
                print(f"Track A (Imputed): {df_imp.shape[0]} rows", file=f)
                print(f"Track B (Dropped): {df_drop.shape[0]} rows", file=f)

            # 3. Preprocessing & Encoding
            print_header("3. Preprocessing & Encoding", file=f)
            sys.__stdout__.write("  3. Preprocessing & Encoding\n")
            X_ta, X_va, y_ta, y_va, log_target, prep_log_a = preprocess_track(
                df_imp, requires_imputation=True, target=cfg['target']
            )

            # Log transform decision
            skew = prep_log_a["skewness"]
            if prep_log_a["log_target"]:
                print(f"  [Log Transform]  Train-set skewness = {skew:+.4f}  "
                      f"(|skew| > {LOG_SKEWNESS_THRESHOLD}) -> np.log1p applied.", file=f)
                print(f"                   Metrics and plots reported in original scale (np.expm1).", file=f)
            else:
                print(f"  [Log Transform]  Train-set skewness = {skew:+.4f}  "
                      f"(|skew| ≤ {LOG_SKEWNESS_THRESHOLD}) -> no transformation.", file=f)

            # Imputation
            if prep_log_a["imputed_cols"]:
                print(f"  [Imputation]     {len(prep_log_a['imputed_cols'])} column(s) with missing values"
                      f" -> median (numeric) / mode (categorical) fill:", file=f)
                print(f"                   {', '.join(prep_log_a['imputed_cols'])}", file=f)
            else:
                print(f"  [Imputation]     No missing values — no imputation required.", file=f)

            # Encoding
            if prep_log_a["encoded_cols"]:
                print(f"  [Encoding]       {len(prep_log_a['encoded_cols'])} categorical column(s) -> "
                      f"{prep_log_a['n_total_dummies']} dummy columns (one-hot, drop_first=True):", file=f)
                print(f"                   {', '.join(prep_log_a['encoded_cols'])}", file=f)
            else:
                print(f"  [Encoding]       No categorical predictors.", file=f)

            # Scaling
            print(f"  [Scaling]        StandardScaler fitted on train fold only -> applied to test (no leakage).\n",
                  file=f)

            if not single_track:
                X_tb, X_vb, y_tb, y_vb, _, _prep_log_b = preprocess_track(
                    df_drop, requires_imputation=False, target=cfg['target']
                )

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
                    preds[name] = y_pred
                    fitted[name] = model
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

            # Back-transform log(y+1) predictions using expm1 for final metrics and plots
            # Ensures RMSE and R² are reported in the original target scale
            if log_target:
                y_va_disp = np.expm1(y_va)
                preds_a_disp = {n: np.expm1(p) for n, p in preds_a.items()}
                results_a = {
                    n: {"rmse": root_mean_squared_error(y_va_disp, preds_a_disp[n]),
                        "r2": r2_score(y_va_disp, preds_a_disp[n])}
                    for n in preds_a
                }
                if single_track:
                    y_vb_disp, preds_b_disp = y_va_disp, preds_a_disp
                    results_b = results_a
                else:
                    y_vb_disp = np.expm1(y_vb)
                    preds_b_disp = {n: np.expm1(p) for n, p in preds_b.items()}
                    results_b = {
                        n: {"rmse": root_mean_squared_error(y_vb_disp, preds_b_disp[n]),
                            "r2": r2_score(y_vb_disp, preds_b_disp[n])}
                        for n in preds_b
                    }
            else:
                y_va_disp, preds_a_disp = y_va, preds_a
                y_vb_disp, preds_b_disp = y_vb, preds_b

            print(f"Final Feature Count: {X_ta.shape[1]} \n", file=f)

            # Optimal hyperparameters
            if "Lasso" in fitted_a:
                alpha_a = fitted_a["Lasso"].alpha
                if single_track:
                    print(f"Optimal Lasso α  — {alpha_a:.4g}", file=f)
                else:
                    alpha_b = fitted_b["Lasso"].alpha
                    print(f"Optimal Lasso α  — Track A: {alpha_a:.4g},  Track B: {alpha_b:.4g}", file=f)

            if "DT (Optimal)" in fitted_a:
                depth_a = fitted_a["DT (Optimal)"].max_depth
                if single_track:
                    print(f"Optimal DT depth — {depth_a}", file=f)
                else:
                    depth_b = fitted_b["DT (Optimal)"].max_depth
                    print(f"Optimal DT depth — Track A: {depth_a},  Track B: {depth_b}", file=f)

            if "Random Forest" in fitted_a:
                n_a = fitted_a["Random Forest"].n_estimators
                if single_track:
                    print(f"Optimal n_estimators — {n_a}", file=f)
                else:
                    n_b = fitted_b["Random Forest"].n_estimators
                    print(f"Optimal n_estimators — Track A: {n_a},  Track B: {n_b}", file=f)

            # 4. Modeling & Visualisation
            print_header("4. Model Evaluation & Visualisation", file=f)
            sys.__stdout__.write("  4. Model Evaluation & Visualisation\n")

            # LR baseline — needed for DT, RF, and Lasso tuning curves
            needs_lr_baseline = any(k in hist_a for k in ("DT", "RF", "Lasso"))
            lr_rmse_a = cv_rmse_lr(X_ta, y_ta) if needs_lr_baseline else None
            lr_rmse_b = cv_rmse_lr(X_tb, y_tb) if (needs_lr_baseline and not single_track) else None

            # Tuning curves — only drawn when the corresponding tuning was performed
            if "Lasso" in hist_a:
                plot_lasso_tuning_curve(hist_a["Lasso"], hist_b["Lasso"],
                                        out_dir / f"Lasso_tuning_{tag}.png",
                                        lr_rmse_a=lr_rmse_a, lr_rmse_b=lr_rmse_b,
                                        single_track=single_track)
                print(f"  [Plot saved]  Lasso_tuning_{tag}.png", file=f)

            if "DT" in hist_a:
                plot_tuning_curve(hist_a["DT"], hist_b["DT"],
                                  out_dir / f"DT_tuning_{tag}.png", lr_rmse_a, lr_rmse_b,
                                  single_track=single_track)
                print(f"  [Plot saved]  DT_tuning_{tag}.png", file=f)

            if "RF" in hist_a:
                plot_tuning_curve(hist_a["RF"], hist_b["RF"],
                                  out_dir / f"RF_tuning_{tag}.png", lr_rmse_a, lr_rmse_b,
                                  x_key="n_estimators", x_label="Number of Trees",
                                  suptitle="CV RMSE vs. Number of Trees  (10-fold CV)",
                                  single_track=single_track)
                print(f"  [Plot saved]  RF_tuning_{tag}.png", file=f)

            # Actual vs. predicted — one figure per fitted model
            for name in [m for m in model_list if m in preds_a_disp]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_actual_vs_predicted(
                    y_va_disp, preds_a_disp[name], y_vb_disp, preds_b_disp[name],
                    name, out_dir / f"{clean}_actual_vs_pred_{tag}.png",
                    single_track=single_track, target=cfg['target']
                )
                print(f"  [Plot saved]  {clean}_actual_vs_pred_{tag}.png", file=f)

            # Residuals — drawn for fitted models among the standard candidates
            for name in [m for m in ["Linear Regression", "Lasso", "DT (Optimal)", "Random Forest"]
                         if m in preds_a_disp]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_residuals(
                    y_va_disp, preds_a_disp[name], y_vb_disp, preds_b_disp[name],
                    name, out_dir / f"{clean}_residuals_{tag}.png",
                    single_track=single_track, target=cfg['target']
                )
                print(f"  [Plot saved]  {clean}_residuals_{tag}.png", file=f)

            # Feature importances — only when the relevant model was fitted
            if "Lasso" in fitted_a:
                plot_lasso_coefficients(
                    fitted_a["Lasso"], fitted_b["Lasso"],
                    X_ta.columns, X_tb.columns,
                    out_dir / f"Lasso_coefficients_{tag}.png",
                    single_track=single_track
                )
                print(f"  [Plot saved]  Lasso_coefficients_{tag}.png", file=f)
                _print_lasso_contrast(
                    fitted_a, fitted_b, X_ta, X_tb,
                    vif_top_feature, vif_top_vif, single_track, f, tag
                )

            if "DT (Optimal)" in fitted_a:
                plot_feature_importance(
                    fitted_a["DT (Optimal)"], fitted_b["DT (Optimal)"],
                    X_ta.columns, X_tb.columns,
                    out_dir / f"DT_feature_importance_{tag}.png",
                    single_track=single_track
                )
                print(f"  [Plot saved]  DT_feature_importance_{tag}.png", file=f)

            if "Random Forest" in fitted_a:
                plot_feature_importance(
                    fitted_a["Random Forest"], fitted_b["Random Forest"],
                    X_ta.columns, X_tb.columns,
                    out_dir / f"RF_feature_importance_{tag}.png",
                    model_name="Random Forest", single_track=single_track
                )
                print(f"  [Plot saved]  RF_feature_importance_{tag}.png", file=f)

            # 5. Residual Normality Diagnostics
            print_header("5. Residual Normality Diagnostics", file=f)
            sys.__stdout__.write("  5. Residual Normality Diagnostics\n")
            sys.stdout = f
            residual_normality_descriptives(
                preds_a_disp, y_va_disp,
                preds_b_disp if not single_track else None,
                y_vb_disp    if not single_track else None,
                single_track=single_track,
                models=model_list,
            )
            sys.stdout = old_stdout

            # Q-Q + histogram normality plots
            for name in [m for m in ["Linear Regression", "Lasso",
                                      "DT (Optimal)", "Random Forest"]
                         if m in preds_a_disp]:
                clean = name.replace(" ", "_").replace("(", "").replace(")", "")
                plot_residual_normality(
                    y_va_disp, preds_a_disp[name],
                    y_vb_disp, preds_b_disp[name],
                    name, out_dir / f"{clean}_normality_{tag}.png",
                    single_track=single_track
                )
                print(f"  [Plot saved]  {clean}_normality_{tag}.png", file=f)

            # 6. Residual Outlier Profile
            print_header("6. Residual Outlier Profile", file=f)
            sys.__stdout__.write("  6. Residual Outlier Profile\n")
            if "Linear Regression" in preds_a_disp:
                sys.stdout = f
                outlier_stats = residual_outlier_profile(
                    df_imp, X_va.index,
                    y_va_disp, preds_a_disp["Linear Regression"],
                    cfg["target"], model_name="Linear Regression"
                )
                sys.stdout = old_stdout
                if outlier_stats:
                    plot_outlier_profile(
                        outlier_stats, "Linear Regression",
                        out_dir / f"outlier_profile_{tag}.png"
                    )
                    print(f"  [Plot saved]  outlier_profile_{tag}.png", file=f)
            else:
                print("  Linear Regression was not fitted — profile skipped.", file=f)

            # 7. Final Comparison Table
            sys.__stdout__.write("  7. Final Comparison Table\n")
            summary_df = create_summary_table(results_a, results_b, single_track=single_track)
            print("\nFinal Model Comparison (Strict 80/20 Test Evaluation):", file=f)
            print(summary_df.to_string(index=False), file=f)

        print(f"Done: {filename} -> {out_dir}/")

if __name__ == "__main__":
    # main(datasets = "winequality-red.csv")
    # main(datasets = {"StudentPerformanceFactors.csv", "Housing.csv"})
    # main(datasets = "Housing.csv")
    main()