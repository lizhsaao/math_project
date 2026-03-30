"""
    Formats the final results into a comparison table and runs residual diagnostics.
"""
import numpy as np
import pandas as pd
from scipy.stats import shapiro


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


def residual_normality_tests(preds_a, y_va, preds_b=None, y_vb=None,
                              single_track=False, models=None):
    """
    Applies the Shapiro-Wilk test to residuals (actual − predicted) for each
    fitted model and prints the results to stdout.

    The null hypothesis H_0 is that residuals are normally distributed.
    A p-value below 0.05 provides evidence against normality at the 5% level.
    The assumption is most relevant for Linear Regression and Lasso; tree-based
    models make no normality assumption on residuals.

    Shapiro-Wilk is accurate for n ≤ 5,000. For larger test sets the W
    statistic is reliable but the p-value becomes conservative.

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

    print(f"\n--- Shapiro-Wilk Residual Normality Test ---")
    print(f"  (H_0: residuals are normally distributed  |  α = 0.05)")

    if single_track:
        print(f"\n  {'Model':<22} {'W':>8}  {'p-value':>10}  {'Result':>12}")
        print(f"  {'-'*58}")
        for name in models:
            if name not in preds_a:
                continue
            resid = y_va_arr - np.asarray(preds_a[name])
            if len(resid) < 3:
                continue
            W, p = shapiro(resid)
            result = "Normal" if p >= 0.05 else "Non-normal"
            print(f"  {name:<22} {W:>8.4f}  {p:>10.4f}  {result:>12}")
    else:
        header = (f"  {'Model':<22} {'W (A)':>8}  {'p (A)':>9}  "
                  f"{'W (B)':>8}  {'p (B)':>9}  {'A':>10}  {'B':>10}")
        print(f"\n{header}")
        print(f"  {'-'*82}")
        for name in models:
            if name not in preds_a:
                continue
            resid_a = y_va_arr - np.asarray(preds_a[name])
            resid_b = y_vb_arr - np.asarray(preds_b[name]) if preds_b else resid_a
            if len(resid_a) < 3:
                continue
            W_a, p_a = shapiro(resid_a)
            W_b, p_b = shapiro(resid_b)
            res_a = "Normal" if p_a >= 0.05 else "Non-normal"
            res_b = "Normal" if p_b >= 0.05 else "Non-normal"
            print(f"  {name:<22} {W_a:>8.4f}  {p_a:>9.4f}  "
                  f"{W_b:>8.4f}  {p_b:>9.4f}  {res_a:>10}  {res_b:>10}")

    print(f"\n  [Normality of residuals is a key assumption for Linear Regression"
          f" and Lasso;\n   tree-based models make no such assumption.]")


def shapiro_wilk_lr(y_true_a, y_pred_lr_a, y_true_b=None, y_pred_lr_b=None,
                    single_track=False, note=""):
    """
    Shapiro-Wilk normality test applied specifically to Linear Regression
    residuals. Prints the W statistic, p-value, and Pass / Fail verdict.

    This is the formal test for the key OLS assumption that errors are
    normally distributed. H_0: residuals ~ Normal.
    Pass if p > 0.05 (fail to reject normality at the 5 % level).

    Parameters
    ----------
    y_true_a, y_pred_lr_a : True and LR-predicted values for Track A.
    y_true_b, y_pred_lr_b : True and LR-predicted values for Track B
                            (ignored when single_track=True).
    single_track          : If True, only Track A is tested.
    note                  : Optional one-line annotation printed below the
                            header (e.g. to indicate which scale was tested).

    Returns
    -------
    dict with keys "W_a", "p_a", "pass_a" (and _b equivalents if two tracks).
    """
    def _run(y_true, y_pred, label):
        resid = np.asarray(y_true) - np.asarray(y_pred)
        W, p  = shapiro(resid)
        result = "Pass" if p > 0.05 else "Fail"
        print(f"  {label:<18}  W = {W:.4f}   p-value = {p:.4f}   [{result}]")
        return W, p, result

    print(f"\n--- Shapiro-Wilk Test: Linear Regression Residuals ---")
    print(f"  (H_0: residuals are normally distributed  |  Pass if p > 0.05)")
    if note:
        print(f"  Note: {note}")
    print(f"  {'-'*60}")

    out = {}
    W_a, p_a, r_a = _run(y_true_a, y_pred_lr_a,
                          "All Data" if single_track else "Track A (Imputed)")
    out.update({"W_a": W_a, "p_a": p_a, "pass_a": r_a == "Pass"})

    if not single_track and y_true_b is not None and y_pred_lr_b is not None:
        W_b, p_b, r_b = _run(y_true_b, y_pred_lr_b, "Track B (Dropped)")
        out.update({"W_b": W_b, "p_b": p_b, "pass_b": r_b == "Pass"})

    return out
