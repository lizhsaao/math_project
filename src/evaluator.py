"""
    Formats the final results into a comparison table and runs residual diagnostics.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def create_summary_table(results_a, results_b, models=None, single_track=False):
    """
    Formats model evaluation results into a comparison table.

    Parameters
    ----------
    results_a, results_b : Dicts mapping model name -> {"rmse", "r2"} for each track.
    models               : Which models to include in the table.
                           - None (default): all models present in results_a,
                             in insertion order.
                           - str: a single model name, e.g. "Lasso".
                           - list[str]: an explicit ordered subset of model names.
    single_track         : If True, renders one set of RMSE/R² columns labelled
                           plainly ("RMSE", "R²") rather than per-track columns.

    Returns
    -------
    pd.DataFrame with model name plus RMSE and R² columns (one set per track,
    or a single set when single_track=True).
    """
    if models is None:
        model_list = list(results_a.keys())
    elif isinstance(models, str):
        model_list = [models]
    else:
        model_list = list(models)

    data = []
    for m in model_list:
        if single_track:
            data.append({
                "Model": m,
                "RMSE":  f"{results_a[m]['rmse']:.4f}",
                "R²":    f"{results_a[m]['r2']:.4f}",
            })
        else:
            data.append({
                "Model":           m,
                "Track A (RMSE)":  f"{results_a[m]['rmse']:.4f}",
                "Track A R²":      f"{results_a[m]['r2']:.4f}",
                "Track B (RMSE)":  f"{results_b[m]['rmse']:.4f}",
                "Track B R²":      f"{results_b[m]['r2']:.4f}",
            })
    return pd.DataFrame(data)


def residual_normality_descriptives(preds_a, y_va, preds_b=None, y_vb=None,
                                     single_track=False, models=None):
    """
    Computes Skewness and Excess Kurtosis for the residuals of each fitted model.

    Normality reference values: Skewness ≈ 0 (symmetric distribution),
    Excess Kurtosis ≈ 0 (mesokurtic — normal tail weight).  Visual
    inspection of Q-Q plots and histograms is the primary normality
    diagnostic; these statistics quantify the extent of any departure.

    Parameters
    ----------
    preds_a, preds_b : Dicts mapping model name -> predicted array (Track A/B).
    y_va, y_vb       : True test-set values (Track A/B). Must be in the same
                       scale as preds (i.e. back-transformed if log was applied).
    single_track     : If True only Track A results are printed.
    models           : Ordered list of model names; defaults to preds_a.keys().
    """
    if models is None:
        models = list(preds_a.keys())

    y_va_arr = np.asarray(y_va)
    y_vb_arr = np.asarray(y_vb) if y_vb is not None else None

    print("\n--- Residual Normality Diagnostics ---")
    print("  Normality is assessed via visual inspection of Q-Q plots and histograms.")
    print("  Skewness/Kurtosis values quantify the extent of departure from normality.")
    print("  Reference: Skewness ≈ 0 (symmetric), Excess Kurtosis ≈ 0 (mesokurtic).\n")

    if single_track:
        print(f"  {'Model':<22} {'Skewness':>9}  {'Ex. Kurtosis':>13}")
        print(f"  {'-'*48}")
        for name in models:
            if name not in preds_a:
                continue
            resid = y_va_arr - np.asarray(preds_a[name])
            s = skew(resid, bias=False)
            k = kurtosis(resid, bias=False)
            print(f"  {name:<22} {s:>+9.4f}  {k:>+13.4f}")
    else:
        print(f"  {'Model':<22} {'Skew(A)':>8}  {'Kurt(A)':>8}  "
              f"{'Skew(B)':>8}  {'Kurt(B)':>8}")
        print(f"  {'-'*60}")
        for name in models:
            if name not in preds_a:
                continue
            resid_a = y_va_arr - np.asarray(preds_a[name])
            resid_b = (y_vb_arr - np.asarray(preds_b[name])
                       if preds_b else resid_a)
            sa = skew(resid_a, bias=False)
            ka = kurtosis(resid_a, bias=False)
            sb = skew(resid_b, bias=False)
            kb = kurtosis(resid_b, bias=False)
            print(f"  {name:<22} {sa:>+8.4f}  {ka:>+8.4f}  "
                  f"{sb:>+8.4f}  {kb:>+8.4f}")

    print("\n  [Normality of residuals is a key assumption for Linear Regression"
          " and Lasso;\n   tree-based models make no such assumption.]")


def residual_outlier_profile(df_original, test_idx, y_true, y_pred,
                              target, model_name, top_pct=0.05):
    """
    Isolates the top `top_pct` fraction of test observations by absolute error
    and cross-references their categorical traits against the full test set.

    Prints a per-column frequency table: population %, outlier %, and Δ pp
    (percentage-point difference). A positive Δ means that subgroup is
    over-represented among the worst predictions — i.e. the model fails
    disproportionately for that category.

    Parameters
    ----------
    df_original : DataFrame with original (pre-encoding) columns — pass df_imp.
    test_idx    : Index labels of test rows in df_original (use X_va.index).
    y_true      : True target values for the test set (array or Series).
    y_pred      : Predicted values for the test set (numpy array).
    target      : Name of the response variable column (excluded from profiling).
    model_name  : Model label used in printed headers.
    top_pct     : Fraction of highest-error observations to isolate (default 0.05).

    Returns
    -------
    dict  {col: {"categories": [...], "delta_pp": [...],
                 "pop_pct": [...], "out_pct": [...]}}
    Returns an empty dict if no categorical predictors exist or the outlier
    group is too small to be reliable.
    """
    df_test = df_original.loc[test_idx]
    cat_cols = [c for c in df_test.select_dtypes(include=["object", "category"]).columns
                if c != target]

    if not cat_cols:
        print(f"\n  No categorical predictors — outlier profile skipped.")
        return {}

    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    cutoff = np.percentile(abs_err, (1 - top_pct) * 100)
    outlier_mask = abs_err >= cutoff
    n_total = len(abs_err)
    n_outlier = int(outlier_mask.sum())

    if n_outlier < 5:
        print(f"\n  Outlier subset too small ({n_outlier} rows) — profile skipped.")
        return {}

    # df_test has the original (non-reset) index from df_original; use it to
    # select outlier rows positionally via the index array.
    outlier_labels = df_test.index[outlier_mask]
    df_outlier = df_test.loc[outlier_labels]

    print(f"\n--- Residual Outlier Profile — {model_name} ---")
    print(f"  Top {top_pct*100:.0f}% absolute errors: "
          f"{n_outlier} / {n_total} test observations  (AE ≥ {cutoff:.4g})")
    print(f"  Δ = (outlier freq) − (population freq)   "
          f"[positive Δ = over-represented in worst predictions]\n")

    outlier_stats = {}
    for col in cat_cols:
        freq_all = df_test[col].value_counts(normalize=True)
        freq_out = df_outlier[col].value_counts(normalize=True)
        all_cats = freq_all.index.union(freq_out.index)

        delta = {cat: (freq_out.get(cat, 0.0) - freq_all.get(cat, 0.0))
                 for cat in all_cats}
        # Sort by |Δ| descending — most deviant categories shown first
        delta_sorted = dict(sorted(delta.items(),
                                   key=lambda x: abs(x[1]), reverse=True))

        print(f"  {col}:")
        print(f"  {'Category':<28}  {'Pop.%':>6}  {'Outlier%':>8}  {'Δ pp':>7}")
        print(f"  {'-'*54}")
        for cat, d in delta_sorted.items():
            pop_pct = freq_all.get(cat, 0.0) * 100
            out_pct = freq_out.get(cat, 0.0) * 100
            flag = "  ◄" if abs(d) >= 0.05 else ""
            print(f"  {str(cat):<28}  {pop_pct:>6.1f}  {out_pct:>8.1f}  "
                  f"{d*100:>+7.1f}{flag}")
        print()

        outlier_stats[col] = {
            "categories": list(delta_sorted.keys()),
            "delta_pp": [v * 100 for v in delta_sorted.values()],
            "pop_pct": [freq_all.get(c, 0.0) * 100 for c in delta_sorted],
            "out_pct": [freq_out.get(c, 0.0) * 100 for c in delta_sorted],
        }

    return outlier_stats
