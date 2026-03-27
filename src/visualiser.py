"""
    Diagnostic plots: EDA (target distribution, correlations), actual vs. predicted,
    residual distributions, tree depth-scan curves, and feature importances.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

# --- Style constants matching the notebook ---
_COL_A   = "steelblue"    # Track A
_COL_B   = "mediumseagreen"        # Track B
_COL_REF = "darkorange"   # reference lines (ideal-fit diagonal, optimal depth vline)
_COL_LR  = "gray"     # LR baseline


def _save(fig, output_path):
    """Saves the figure at 150 dpi and closes it."""
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_exam_score_distribution(df, target, output_path):
    """
    Histogram of the target variable with a Normal distribution overlay.

    Parameters
    ----------
    df          : DataFrame containing the target column.
    target      : Name of the target/response variable column.
    output_path : File path to save the figure.
    """
    mu  = df[target].mean()
    sig = df[target].std()
    x   = np.linspace(mu - 4 * sig, mu + 4 * sig, 300)
    pdf = (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    fig, ax = subplots(figsize=(7, 4))
    ax.hist(df[target], bins=20, density=True,
            color=_COL_A, alpha=0.7, edgecolor="white", label="Observed")
    ax.plot(x, pdf, "r-", linewidth=2,
            label=f"Normal fit (μ = {mu:.2f}, σ = {sig:.2f})")
    ax.set_title(f"Distribution of {target}", fontsize=13)
    ax.set_xlabel(target)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_path)


def plot_correlation_with_target(df_a, df_b, target, output_path, single_track=False):
    """
    Horizontal bar charts of Pearson correlations between each numerical
    predictor and the target variable. Side-by-side for two tracks; single
    panel when single_track=True.

    Parameters
    ----------
    df_a, df_b   : DataFrames for Track A (imputed) and Track B (dropped).
    target       : Name of the target/response variable column.
    output_path  : File path to save the figure.
    single_track : If True, renders one panel (Track A data only).
    """
    corr_a = df_a.corr(numeric_only=True)[target].drop(target).sort_values()
    corr_b = df_b.corr(numeric_only=True)[target].drop(target).sort_values()

    if single_track:
        fig, ax = subplots(1, 1, figsize=(7, 4))
        _tracks = [(ax, corr_a, _COL_A, "All Data")]
    else:
        fig, axes = subplots(1, 2, figsize=(12, 4))
        _tracks = list(zip(axes, [corr_a, corr_b], [_COL_A, _COL_B],
                           ["Track A (Imputed)", "Track B (Dropped)"]))

    for ax, corr, col, title in _tracks:
        ax.barh(corr.index, corr.values, color=col, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"Pearson r with {target}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Correlation with {target}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_actual_vs_predicted(y_true_a, y_pred_a, y_true_b, y_pred_b, model_name, output_path,
                             single_track=False, target="value"):
    """
    Actual vs. predicted scatter plots. Side-by-side for two tracks; single
    panel when single_track=True.

    Parameters
    ----------
    y_true_a, y_pred_a : True and predicted values for Track A.
    y_true_b, y_pred_b : True and predicted values for Track B.
    model_name         : Model label used in the figure title.
    output_path        : File path to save the figure.
    single_track       : If True, renders one panel (Track A data only).
    target             : Name of the response variable (used for axis labels).
    """
    if single_track:
        fig, ax = subplots(1, 1, figsize=(6, 5))
        _tracks = [(ax, y_true_a, y_pred_a, _COL_A, "All Data")]
    else:
        fig, axes = subplots(1, 2, figsize=(12, 5), sharey=True)
        _tracks = list(zip(axes, [y_true_a, y_true_b], [y_pred_a, y_pred_b],
                           [_COL_A, _COL_B], ["Track A (Imputed)", "Track B (Dropped)"]))

    _label = target.replace("_", " ")
    for ax, y_true, y_pred, col, title in _tracks:
        ax.scatter(y_true, y_pred, color=col, alpha=0.4, s=14, edgecolors="none")
        lo = min(y_true.min(), y_pred.min()) - 0.5
        hi = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "--", color=_COL_REF, linewidth=1.5, label="Ideal fit")
        ax.set_xlabel(f"Actual {_label}")
        ax.set_ylabel(f"Predicted {_label}")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Actual vs. Predicted — {model_name}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_tuning_curve(history_a, history_b, output_path,
                      lr_rmse_a=None, lr_rmse_b=None,
                      x_key="depth", x_label="Tree Depth",
                      suptitle="CV RMSE vs. Tree Depth  (10-fold CV)",
                      n_folds=10, single_track=False):
    """
    CV RMSE tuning curve with ±1 s.d. band, 1-SE threshold line, and optional
    LR baseline. Side-by-side for two tracks; single panel when single_track=True.

    Parameters
    ----------
    history_a, history_b : Lists of {"<x_key>", "rmse", "std"} dicts (one per track).
    output_path          : File path to save the figure.
    lr_rmse_a, lr_rmse_b : Optional LR CV RMSE drawn as a reference line.
    x_key                : Key in history dicts used for the x-axis (default "depth").
    x_label              : x-axis label string (default "Tree Depth").
    suptitle             : Figure-level title (default matches DT depth scan).
    n_folds              : Number of CV folds used; needed to compute SE (default 10).
    single_track         : If True, renders one panel (Track A data only).
    """
    if single_track:
        fig, ax = subplots(1, 1, figsize=(7, 5))
        _tracks = [(ax, history_a, _COL_A, lr_rmse_a, "All Data")]
    else:
        fig, axes = subplots(1, 2, figsize=(13, 5), sharey=True)
        _tracks = list(zip(axes, [history_a, history_b], [_COL_A, _COL_B],
                           [lr_rmse_a, lr_rmse_b], ["Track A (Imputed)", "Track B (Dropped)"]))

    for ax, history, col, lr_rmse, title in _tracks:
        df       = pd.DataFrame(history)
        x_vals   = df[x_key].values
        rmse_arr = df["rmse"].values
        std_arr  = df["std"].values

        # 1-SE rule: horizontal line at min RMSE; interpolate the exact x where
        # the lower ±1 s.d. boundary (rmse - std) crosses that line going left.
        min_idx   = int(np.argmin(rmse_arr))
        threshold = float(rmse_arr[min_idx])
        lower_arr = rmse_arr - std_arr

        # Linear interpolation between the last point above and first point below
        cross_x = float(x_vals[0])   # fallback: first point already below
        for i in range(len(x_vals) - 1):
            if lower_arr[i] > threshold and lower_arr[i + 1] <= threshold:
                t = (threshold - lower_arr[i]) / (lower_arr[i + 1] - lower_arr[i])
                cross_x = float(x_vals[i] + t * (x_vals[i + 1] - x_vals[i]))
                break

        ax.plot(x_vals, rmse_arr, "o-", color=col, markersize=5, label="CV RMSE")
        ax.fill_between(x_vals, lower_arr, rmse_arr + std_arr,
                        alpha=0.15, color=col, label="±1 s.d.")
        ax.axhline(threshold, linestyle="--", color="crimson", linewidth=1.2,
                   label=f"Min RMSE ({threshold:.3f})")
        ax.axvline(cross_x, linestyle="--", color=_COL_REF,
                   label=f"1-SE crossing = {cross_x:.1f}")

        if lr_rmse is not None:
            ax.axhline(lr_rmse, linestyle=":", color=_COL_LR, linewidth=1.5,
                       label=f"LR baseline  ({lr_rmse:.3f})")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel("CV RMSE")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_residuals(y_true_a, y_pred_a, y_true_b, y_pred_b, model_name, output_path,
                   single_track=False, target="value"):
    """
    Residuals vs. predicted scatter plots. Side-by-side for two tracks; single
    panel when single_track=True.

    Parameters
    ----------
    y_true_a, y_pred_a : True and predicted values for Track A.
    y_true_b, y_pred_b : True and predicted values for Track B.
    model_name         : Model label used in the figure title.
    output_path        : File path to save the figure.
    single_track       : If True, renders one panel (Track A data only).
    target             : Name of the response variable (used for axis label).
    """
    if single_track:
        fig, ax = subplots(1, 1, figsize=(7, 4))
        _tracks = [(ax, y_true_a, y_pred_a, _COL_A, "All Data")]
    else:
        fig, axes = subplots(1, 2, figsize=(12, 4), sharey=True)
        _tracks = list(zip(axes, [y_true_a, y_true_b], [y_pred_a, y_pred_b],
                           [_COL_A, _COL_B], ["Track A (Imputed)", "Track B (Dropped)"]))

    _label = target.replace("_", " ")
    for ax, y_true, y_pred, col, title in _tracks:
        resid = y_true.values - y_pred
        ax.scatter(y_pred, resid, color=col, alpha=0.4, s=14, edgecolors="none")
        ax.axhline(0, linestyle=":", color=_COL_REF, linewidth=1.5)
        ax.set_xlabel(f"Predicted {_label}")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Residuals vs. Predicted — {model_name}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_feature_importance(model_a, model_b, feat_names_a, feat_names_b, output_path,
                            model_name="DT (Optimal)", single_track=False):
    """
    Horizontal bar charts of the top 10 feature importances. Side-by-side for
    two tracks; single panel when single_track=True.

    Parameters
    ----------
    model_a, model_b           : Fitted tree estimators with feature_importances_.
    feat_names_a, feat_names_b : Column names from each track's training matrix.
    output_path                : File path to save the figure.
    model_name                 : Model label used in the figure title (default "DT (Optimal)").
    single_track               : If True, renders one panel (Track A data only).
    """
    if single_track:
        fig, ax = subplots(1, 1, figsize=(8, 6))
        _tracks = [(ax, model_a, feat_names_a, _COL_A, "All Data")]
    else:
        fig, axes = subplots(1, 2, figsize=(14, 6))
        _tracks = list(zip(axes, [model_a, model_b], [feat_names_a, feat_names_b],
                           [_COL_A, _COL_B], ["Track A (Imputed)", "Track B (Dropped)"]))

    for ax, model, names, col, title in _tracks:
        importances = pd.Series(model.feature_importances_, index=names)
        top10 = importances.sort_values(ascending=True).tail(10)

        # Horizontal bars; ascending sort + tail gives longest bar at the top
        ax.barh(top10.index, top10.values, color=col, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Top 10 Feature Importances — {model_name}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_lasso_tuning_curve(history_a, history_b, output_path,
                            lr_rmse_a=None, lr_rmse_b=None, n_folds=10,
                            single_track=False):
    """
    CV RMSE vs. Lasso alpha on a log x-axis with ±1 s.d. band and 1-SE selection
    line. Side-by-side for two tracks; single panel when single_track=True.

    Parameters
    ----------
    history_a, history_b : Lists of {"alpha", "rmse", "std"} dicts (one per track).
    output_path          : File path to save the figure.
    lr_rmse_a, lr_rmse_b : Optional LR CV RMSE reference lines (one per track).
    n_folds              : Number of CV folds used (default 10).
    single_track         : If True, renders one panel (Track A data only).
    """
    if single_track:
        fig, ax = subplots(1, 1, figsize=(7, 5))
        _tracks = [(ax, history_a, _COL_A, lr_rmse_a, "All Data")]
    else:
        fig, axes = subplots(1, 2, figsize=(13, 5), sharey=True)
        _tracks = list(zip(axes, [history_a, history_b], [_COL_A, _COL_B],
                           [lr_rmse_a, lr_rmse_b], ["Track A (Imputed)", "Track B (Dropped)"]))

    for ax, history, col, lr_rmse, title in _tracks:
        df       = pd.DataFrame(history)
        x_vals   = df["alpha"].values
        rmse_arr = df["rmse"].values
        std_arr  = df["std"].values

        min_idx   = int(np.argmin(rmse_arr))
        threshold = float(rmse_arr[min_idx])
        lower_arr = rmse_arr - std_arr

        # 1-SE rule for Lasso: scan right to left for the RIGHTMOST crossing
        cross_x = float(x_vals[min_idx])
        for i in range(len(x_vals) - 1, 0, -1):
            if lower_arr[i] > threshold and lower_arr[i - 1] <= threshold:
                t = (threshold - lower_arr[i - 1]) / (lower_arr[i] - lower_arr[i - 1])
                log_cross = np.log10(x_vals[i - 1]) + t * (
                    np.log10(x_vals[i]) - np.log10(x_vals[i - 1])
                )
                cross_x = float(10 ** log_cross)
                break

        ax.plot(x_vals, rmse_arr, "o-", color=col, markersize=4, label="CV RMSE")
        ax.fill_between(x_vals, lower_arr, rmse_arr + std_arr,
                        alpha=0.15, color=col, label="±1 s.d.")
        
        # Min RMSE line (Red)
        ax.axhline(threshold, linestyle="--", color="crimson", linewidth=1.2,
                   label=f"Min RMSE ({threshold:.3f})")
        
        # 1-SE crossing (Orange)
        ax.axvline(cross_x, linestyle="--", color=_COL_REF,
                   label=f"1-SE crossing = {cross_x:.3g}")

        # ADDED: LR baseline (Green dotted)
        if lr_rmse is not None:
            ax.axhline(lr_rmse, linestyle=":", color=_COL_LR, linewidth=1.5,
                       label=f"LR baseline  ({lr_rmse:.3f})")

        ax.set_xscale("log")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("α  (log scale)")
        ax.set_ylabel("CV RMSE")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
    fig.suptitle("CV RMSE vs. Lasso α  (10-fold CV)", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_lasso_coefficients(model_a, model_b, feat_names_a, feat_names_b, output_path,
                            single_track=False):
    """
    Horizontal bar charts of Lasso coefficients (non-zero only). Side-by-side
    for two tracks; single panel when single_track=True.

    Parameters
    ----------
    model_a, model_b           : Fitted Lasso estimators with a .coef_ attribute.
    feat_names_a, feat_names_b : Column names from each track's training matrix.
    output_path                : File path to save the figure.
    single_track               : If True, renders one panel (Track A data only).
    """
    if single_track:
        fig, ax = subplots(1, 1, figsize=(8, 6))
        _tracks = [(ax, model_a, feat_names_a, "All Data")]
    else:
        fig, axes = subplots(1, 2, figsize=(14, 6))
        _tracks = list(zip(axes, [model_a, model_b], [feat_names_a, feat_names_b],
                           ["Track A (Imputed)", "Track B (Dropped)"]))

    for ax, model, names, title in _tracks:
        coefs = pd.Series(model.coef_, index=names)
        non_zero = coefs[coefs != 0].sort_values()

        colors = [_COL_A if v > 0 else _COL_B for v in non_zero.values]
        ax.barh(non_zero.index, non_zero.values, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{title} ({len(non_zero)} non-zero / {len(coefs)} features)",
            fontsize=11
        )
        ax.set_xlabel("Coefficient")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Lasso Coefficients (non-zero only)", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)
