"""
    Diagnostic plots: EDA (target distribution, correlations), actual vs. predicted,
    residual distributions, hyperparameter tuning curves, and feature importances.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from scipy.stats import probplot
from pathlib import Path

# --- Style constants matching the notebook ---
_COL_A   = "steelblue"    # Track A
_COL_B   = "mediumseagreen"    # Track B
_COL_REF = "darkorange"   # reference lines (ideal-fit diagonal, optimal depth vline)
_COL_LR  = "gray"     # LR baseline


def _save(fig, output_path):
    """
    Saves the matplotlib figure at 150 dpi and closes it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    output_path : str or pathlib.Path
        File path where the image will be saved.
    """
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _get_track_path(output_path, track_suffix):
    """
    Injects a track suffix (e.g., '_TrackA') into the filename before the extension.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Original file path.
    track_suffix : str
        Suffix to append (e.g., '_TrackA').
    """
    if not track_suffix:
        return output_path
    
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}{track_suffix}{p.suffix}"))


def plot_exam_score_distribution(df, target, output_path, bin_strategy="fd"):
    """
    Plots a histogram of the target variable with a Normal distribution overlay.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the target column.
    target : str
        Name of the target variable.
    output_path : str or pathlib.Path
        File path to save the figure.
    bin_strategy : str, optional
        Binning method: "unit", "integer", or "fd" (default is "fd").
    """
    data = df[target].dropna()

    if bin_strategy == "unit":
        lo, hi = int(data.min()), int(data.max())
        bins = range(lo, hi + 2)          # +2 so the last bar is fully drawn

    elif bin_strategy == "integer":
        bins = np.arange(data.min() - 0.5, data.max() + 1.5, 1.0)

    else:  # "fd" — Freedman-Diaconis
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            bins = 20
        else:
            bin_width = 2 * iqr * len(data) ** (-1 / 3)
            bins = max(5, int(np.ceil((data.max() - data.min()) / bin_width)))

    mu  = data.mean()
    sig = data.std()
    x   = np.linspace(mu - 4 * sig, mu + 4 * sig, 300)
    pdf = (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    fig, ax = subplots(figsize=(7, 4))
    ax.hist(data, bins=bins, density=True,
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
    Plots horizontal bar charts of Pearson correlations between predictors and target.

    Parameters
    ----------
    df_a, df_b : pandas.DataFrame
        DataFrames for Track A and Track B.
    target : str
        Name of the target variable.
    output_path : str or pathlib.Path
        File path to save the figure.
    single_track : bool, optional
        If True, renders only Track A.
    """
    corr_a = df_a.corr(numeric_only=True)[target].drop(target).sort_values()
    
    if single_track:
        _tracks = [(corr_a, _COL_A, "Correlation with Target", "")]
    else:
        corr_b = df_b.corr(numeric_only=True)[target].drop(target).sort_values()
        _tracks = [
            (corr_a, _COL_A, "Correlation with Target (Track A: Imputed)", "_TrackA"),
            (corr_b, _COL_B, "Correlation with Target (Track B: Dropped)", "_TrackB")
        ]

    for corr, col, title, suffix in _tracks:
        fig, ax = subplots(figsize=(7, 4))
        ax.barh(corr.index, corr.values, color=col, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"Pearson r with {target}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))


def plot_actual_vs_predicted(y_true_a, y_pred_a, y_true_b, y_pred_b, model_name, output_path,
                             single_track=False, target="value"):
    """
    Plots predicted vs. actual scatter charts with an ideal-fit reference line.

    Parameters
    ----------
    y_true_a, y_pred_a : array-like
        True and predicted values for Track A.
    y_true_b, y_pred_b : array-like
        True and predicted values for Track B.
    model_name : str
        Model name for the figure title.
    output_path : str or pathlib.Path
        File path to save the figure.
    single_track : bool, optional
        If True, renders only Track A.
    target : str, optional
        Target variable name for axis labels.
    """
    if single_track:
        _tracks = [(y_true_a, y_pred_a, _COL_A, f"Actual vs. Predicted — {model_name}", "")]
    else:
        _tracks = [
            (y_true_a, y_pred_a, _COL_A, f"Actual vs. Predicted — {model_name} (Track A)", "_TrackA"),
            (y_true_b, y_pred_b, _COL_B, f"Actual vs. Predicted — {model_name} (Track B)", "_TrackB")
        ]

    _label = target.replace("_", " ")
    for y_true, y_pred, col, title, suffix in _tracks:
        fig, ax = subplots(figsize=(6, 5))
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
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))


def plot_tuning_curve(history_a, history_b, output_path,
                      lr_rmse_a=None, lr_rmse_b=None,
                      x_key="depth", x_label="Tree Depth",
                      suptitle="CV RMSE vs. Tree Depth  (10-fold CV)",
                      n_folds=10, single_track=False):
    """
    Plots a CV RMSE tuning curve with ±1 s.e. bands and a 1-SE threshold line.
    
    Parameters
    ----------
    history_a, history_b : list of dict
        Tuning histories for Tracks A and B.
    output_path : str or pathlib.Path
        File path to save the figure.
    lr_rmse_a, lr_rmse_b : float, optional
        Linear Regression baseline RMSEs.
    x_key : str, optional
        Dictionary key for x-axis values.
    x_label : str, optional
        Label for the x-axis.
    suptitle : str, optional
        Figure title.
    n_folds : int, optional
        Number of CV folds.
    single_track : bool, optional
        If True, renders only Track A.
    """
    if single_track:
        _tracks = [(history_a, _COL_A, lr_rmse_a, suptitle, "")]
    else:
        _tracks = [
            (history_a, _COL_A, lr_rmse_a, f"{suptitle} (Track A)", "_TrackA"),
            (history_b, _COL_B, lr_rmse_b, f"{suptitle} (Track B)", "_TrackB")
        ]

    for history, col, lr_rmse, title, suffix in _tracks:
        fig, ax = subplots(figsize=(7, 5))
        df       = pd.DataFrame(history)
        x_vals   = df[x_key].values
        rmse_arr = df["rmse"].values
        se_arr  = df["se"].values

        min_idx   = int(np.argmin(rmse_arr))
        threshold = float(rmse_arr[min_idx] + se_arr[min_idx])
        lower_arr = rmse_arr - se_arr

        cross_x = float(x_vals[0])
        for x, rmse in zip(x_vals, rmse_arr):
            if rmse <= threshold:
                cross_x = float(x)
                break

        ax.plot(x_vals, rmse_arr, "o-", color=col, markersize=5, label="CV RMSE")
        ax.fill_between(x_vals, lower_arr, rmse_arr + se_arr,
                        alpha=0.15, color=col, label="±1 s.e.")
        
        ax.axhline(threshold, linestyle="--", color="crimson", linewidth=1.2,
                label=f"1-SE threshold ({threshold:.3f})")
        ax.axvline(cross_x, linestyle="--", color=_COL_REF,
                label=f"1-SE selected = {cross_x:.1f}")

        if lr_rmse is not None:
            ax.axhline(lr_rmse, linestyle=":", color=_COL_LR, linewidth=1.5,
                       label=f"LR baseline  ({lr_rmse:.3f})")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel("CV RMSE")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))


def plot_residuals(y_true_a, y_pred_a, y_true_b, y_pred_b, model_name, output_path,
                   single_track=False, target="value"):
    """
    Plots residuals against predicted values to visually check for homoscedasticity.

    Parameters
    ----------
    y_true_a, y_pred_a : array-like
        True and predicted values for Track A.
    y_true_b, y_pred_b : array-like
        True and predicted values for Track B.
    model_name : str
        Model name for the figure title.
    output_path : str or pathlib.Path
        File path to save the figure.
    single_track : bool, optional
        If True, renders only Track A.
    target : str, optional
        Target variable name for axis labels.
    """
    if single_track:
        _tracks = [(y_true_a, y_pred_a, _COL_A, f"Residuals vs. Predicted — {model_name}", "")]
    else:
        _tracks = [
            (y_true_a, y_pred_a, _COL_A, f"Residuals vs. Predicted — {model_name} (Track A)", "_TrackA"),
            (y_true_b, y_pred_b, _COL_B, f"Residuals vs. Predicted — {model_name} (Track B)", "_TrackB")
        ]

    _label = target.replace("_", " ")
    for y_true, y_pred, col, title, suffix in _tracks:
        fig, ax = subplots(figsize=(7, 4))
        resid = y_true.values - y_pred
        ax.scatter(y_pred, resid, color=col, alpha=0.4, s=14, edgecolors="none")
        ax.axhline(0, linestyle=":", color=_COL_REF, linewidth=1.5)
        ax.set_xlabel(f"Predicted {_label}")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))


def plot_feature_importance(model_a, model_b, feat_names_a, feat_names_b, output_path,
                            model_name="DT (Optimal)", single_track=False):
    """
    Plots horizontal bar charts of the top 10 feature importances.

    Parameters
    ----------
    model_a, model_b : estimator objects
        Fitted tree estimators with a `feature_importances_` attribute.
    feat_names_a, feat_names_b : list of str
        Feature names for Tracks A and B.
    output_path : str or pathlib.Path
        File path to save the figure.
    model_name : str, optional
        Model name for the figure title.
    single_track : bool, optional
        If True, renders only Track A.
    """
    if single_track:
        _tracks = [(model_a, feat_names_a, _COL_A, f"Top 10 Feature Importances — {model_name}", "")]
    else:
        _tracks = [
            (model_a, feat_names_a, _COL_A, f"Top 10 Feature Importances — {model_name} (Track A)", "_TrackA"),
            (model_b, feat_names_b, _COL_B, f"Top 10 Feature Importances — {model_name} (Track B)", "_TrackB")
        ]

    for model, names, col, title, suffix in _tracks:
        fig, ax = subplots(figsize=(8, 6))
        importances = pd.Series(model.feature_importances_, index=names)
        top10 = importances.sort_values(ascending=True).tail(10)

        ax.barh(top10.index, top10.values, color=col, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))


def plot_lasso_tuning_curve(history_a, history_b, output_path,
                            lr_rmse_a=None, lr_rmse_b=None, n_folds=10,
                            single_track=False):
    """
    Plots CV RMSE vs. Lasso alpha on a log scale with a 1-SE selection line.

    Parameters
    ----------
    history_a, history_b : list of dict
        Tuning histories for Tracks A and B.
    output_path : str or pathlib.Path
        File path to save the figure.
    lr_rmse_a, lr_rmse_b : float, optional
        Linear Regression baseline RMSEs.
    n_folds : int, optional
        Number of CV folds.
    single_track : bool, optional
        If True, renders only Track A.
    """
    if single_track:
        _tracks = [(history_a, _COL_A, lr_rmse_a, "CV RMSE vs. Lasso α", "")]
    else:
        _tracks = [
            (history_a, _COL_A, lr_rmse_a, "CV RMSE vs. Lasso α (Track A)", "_TrackA"),
            (history_b, _COL_B, lr_rmse_b, "CV RMSE vs. Lasso α (Track B)", "_TrackB")
        ]

    for history, col, lr_rmse, title, suffix in _tracks:
        fig, ax = subplots(figsize=(7, 5))
        df       = pd.DataFrame(history)
        x_vals   = df["alpha"].values
        rmse_arr = df["rmse"].values
        se_arr  = df["se"].values

        min_idx   = int(np.argmin(rmse_arr))
        threshold = float(rmse_arr[min_idx] + se_arr[min_idx])
        lower_arr = rmse_arr - se_arr

        cross_x = float(x_vals[min_idx])
        for x, rmse in zip(x_vals[::-1], rmse_arr[::-1]):
            if rmse <= threshold:
                cross_x = float(x)
                break

        ax.plot(x_vals, rmse_arr, "o-", color=col, markersize=4, label="CV RMSE")
        ax.fill_between(x_vals, lower_arr, rmse_arr + se_arr,
                        alpha=0.15, color=col, label="±1 s.e.")
        
        ax.axhline(threshold, linestyle="--", color="crimson", linewidth=1.2,
                label=f"1-SE threshold ({threshold:.3f})")
        ax.axvline(cross_x, linestyle="--", color=_COL_REF,
                label=f"1-SE selected = {cross_x:.3g}")

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
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))


def plot_lasso_coefficients(model_a, model_b, feat_names_a, feat_names_b, output_path,
                            single_track=False):
    """
    Plots horizontal bar charts of non-zero Lasso coefficients.

    Parameters
    ----------
    model_a, model_b : sklearn.linear_model.Lasso
        Fitted Lasso estimators.
    feat_names_a, feat_names_b : list of str
        Feature names for Tracks A and B.
    output_path : str or pathlib.Path
        File path to save the figure.
    single_track : bool, optional
        If True, renders only Track A.
    """
    if single_track:
        _tracks = [(model_a, feat_names_a, "Lasso Coefficients (All Data)", "")]
    else:
        _tracks = [
            (model_a, feat_names_a, "Lasso Coefficients (Track A)", "_TrackA"),
            (model_b, feat_names_b, "Lasso Coefficients (Track B)", "_TrackB")
        ]

    for model, names, title, suffix in _tracks:
        fig, ax = subplots(figsize=(8, 6))
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
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))

def plot_outlier_profile(outlier_stats, model_name, output_path):
    """
    Plots a diverging bar chart comparing outlier subgroup frequencies to the test set.

    Parameters
    ----------
    outlier_stats : dict
        Categorical frequency diffs returned by `residual_outlier_profile()`.
    model_name : str
        Model name for the figure title.
    output_path : str or pathlib.Path
        File path to save the figure.
    """
    if not outlier_stats:
        return

    cols = list(outlier_stats.keys())
    n = len(cols)
    ncols_plot = min(n, 4)
    nrows_plot = (n + ncols_plot - 1) // ncols_plot

    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                              figsize=(4.5 * ncols_plot, 3.5 * nrows_plot))
    # Always work with a flat list, even for a single subplot
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax    = axes_flat[i]
        cats  = outlier_stats[col]["categories"]
        delta = outlier_stats[col]["delta_pp"]

        # Sort by Δ ascending — most negative at bottom, most positive at top
        pairs = sorted(zip(delta, cats))
        delta_s  = [p[0] for p in pairs]
        cats_s   = [str(p[1]) for p in pairs]
        colors_s = [_COL_REF if d > 0 else _COL_A for d in delta_s]

        ax.barh(range(len(cats_s)), delta_s, color=colors_s, alpha=0.85)
        ax.set_yticks(range(len(cats_s)))
        ax.set_yticklabels(cats_s, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Δ frequency (pp)", fontsize=9)
        ax.set_title(col, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide any unused subplot panels in the grid
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Residual Outlier Profile — {model_name}\n"
        f"(top 5% AE vs. full test set  |  "
        f"orange = over-represented in worst predictions)",
        fontsize=10
    )
    plt.tight_layout()
    _save(fig, output_path)


def plot_residual_normality(y_true_a, y_pred_a, y_true_b, y_pred_b,
                             model_name, output_path, single_track=False):
    """
    Combined Q-Q plot and residual histogram with Normal overlay.

    Left column  : Q-Q plot — residual quantiles vs theoretical Normal
                   quantiles with 45-degree reference line.
    Right column : Histogram of residuals with fitted Normal PDF overlay
                   to visually assess symmetry and bell-shape.

    Single-track renders one row; dual-track renders two rows (A/B).

    Parameters
    ----------
    y_true_a, y_pred_a : array-like
        True and predicted values for Track A.
    y_true_b, y_pred_b : array-like
        True and predicted values for Track B.
    model_name : str
        Model name for the figure title.
    output_path : str or pathlib.Path
        File path to save the figure.
    single_track : bool, optional
        If True, renders only Track A.
    """
    if single_track:
        _tracks = [("All Data", y_true_a, y_pred_a, _COL_A, "")]
    else:
        _tracks = [
            ("Track A (Imputed)", y_true_a, y_pred_a, _COL_A, "_TrackA"),
            ("Track B (Dropped)", y_true_b, y_pred_b, _COL_B, "_TrackB")
        ]

    for label, y_true, y_pred, col, suffix in _tracks:
        fig, axes = subplots(1, 2, figsize=(10, 4))
        resid = np.asarray(y_true) - np.asarray(y_pred)

        # --- Q-Q plot ---
        ax_qq = axes[0]
        (osm, osr), (slope, intercept, _) = probplot(resid, dist="norm")
        ax_qq.scatter(osm, osr, s=12, alpha=0.5, color=col, edgecolors="none")
        line_x = np.array([osm.min(), osm.max()])
        ax_qq.plot(line_x, slope * line_x + intercept,
                   "--", color=_COL_REF, linewidth=1.5, label="45° ref")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Residual Quantiles")
        ax_qq.set_title(f"Q-Q", fontsize=10)
        ax_qq.legend(fontsize=8)
        ax_qq.spines["top"].set_visible(False)
        ax_qq.spines["right"].set_visible(False)

        # --- Residual histogram ---
        ax_hist = axes[1]
        ax_hist.hist(resid, bins="fd", density=True,
                     color=col, alpha=0.7, edgecolor="white")
        mu, sig = resid.mean(), resid.std()
        x = np.linspace(mu - 4 * sig, mu + 4 * sig, 300)
        pdf = (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        ax_hist.plot(x, pdf, "r-", linewidth=2, label=f"N({mu:.2f}, {sig:.2f}²)")
        ax_hist.set_xlabel("Residual")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Histogram", fontsize=10)
        ax_hist.legend(fontsize=8)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

        fig.suptitle(f"Residual Normality — {model_name} ({label})", fontsize=13)
        plt.tight_layout()
        _save(fig, _get_track_path(output_path, suffix))