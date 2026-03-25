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
_COL_B   = "coral"        # Track B
_COL_REF = "darkorange"   # reference lines (ideal-fit diagonal, optimal depth vline)
_COL_LR  = "seagreen"     # LR baseline


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


def plot_correlation_with_target(df_a, df_b, target, output_path):
    """
    Side-by-side horizontal bar charts of Pearson correlations between each
    numerical predictor and the target variable, for Track A and Track B.

    Parameters
    ----------
    df_a, df_b  : DataFrames for Track A (imputed) and Track B (dropped).
    target      : Name of the target/response variable column.
    output_path : File path to save the figure.
    """
    corr_a = df_a.corr(numeric_only=True)[target].drop(target).sort_values()
    corr_b = df_b.corr(numeric_only=True)[target].drop(target).sort_values()

    fig, axes = subplots(1, 2, figsize=(12, 4))

    for ax, corr, col, title in zip(
        axes,
        [corr_a, corr_b],
        [_COL_A, _COL_B],
        ["Track A (Imputed)", "Track B (Dropped)"],
    ):
        ax.barh(corr.index, corr.values, color=col, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"Pearson r with {target}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Correlation with {target}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_actual_vs_predicted(y_true_a, y_pred_a, y_true_b, y_pred_b, model_name, output_path):
    """
    Side-by-side actual vs. predicted scatter plots for Track A and Track B.

    Parameters
    ----------
    y_true_a, y_pred_a : True and predicted values for Track A.
    y_true_b, y_pred_b : True and predicted values for Track B.
    model_name         : Model label used in the figure title.
    output_path        : File path to save the figure.
    """
    fig, axes = subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, y_true, y_pred, col, title in zip(
        axes,
        [y_true_a, y_true_b],
        [y_pred_a, y_pred_b],
        [_COL_A, _COL_B],
        ["Track A (Imputed)", "Track B (Dropped)"],
    ):
        ax.scatter(y_true, y_pred, color=col, alpha=0.4, s=14, edgecolors="none")
        lo = min(y_true.min(), y_pred.min()) - 0.5
        hi = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "--", color=_COL_REF, linewidth=1.5, label="Ideal fit")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Actual vs. Predicted — {model_name}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_tuning_curve(history_a, history_b, output_path, lr_rmse_a=None, lr_rmse_b=None):
    """
    Side-by-side CV RMSE vs. tree depth with ±1 s.d. band and optional LR baseline.

    Parameters
    ----------
    history_a, history_b : Lists of {"depth", "rmse", "std"} dicts (one per track).
    output_path          : File path to save the figure.
    lr_rmse_a, lr_rmse_b : Optional LR hold-out RMSE drawn as a reference line.
    """
    fig, axes = subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, history, col, lr_rmse, title in zip(
        axes,
        [history_a, history_b],
        [_COL_A, _COL_B],
        [lr_rmse_a, lr_rmse_b],
        ["Track A (Imputed)", "Track B (Dropped)"],
    ):
        df      = pd.DataFrame(history)
        depths   = df["depth"].values
        rmse_arr = df["rmse"].values
        std_arr  = df["std"].values
        opt_d    = int(df.loc[df["rmse"].idxmin(), "depth"])

        ax.plot(depths, rmse_arr, "o-", color=col, markersize=5, label="CV RMSE")
        ax.fill_between(depths, rmse_arr - std_arr, rmse_arr + std_arr,
                        alpha=0.15, color=col, label="±1 s.d.")
        ax.axvline(opt_d, linestyle="--", color=_COL_REF,
                   label=f"Optimal depth = {opt_d}")

        if lr_rmse is not None:
            ax.axhline(lr_rmse, linestyle=":", color=_COL_LR, linewidth=1.5,
                       label=f"LR baseline  ({lr_rmse:.3f})")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Tree Depth")
        ax.set_ylabel("CV RMSE")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("CV RMSE vs. Tree Depth  (10-fold CV)", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_residuals(y_true_a, y_pred_a, y_true_b, y_pred_b, model_name, output_path):
    """
    Side-by-side residuals vs. predicted plots for Track A and Track B.

    Parameters
    ----------
    y_true_a, y_pred_a : True and predicted values for Track A.
    y_true_b, y_pred_b : True and predicted values for Track B.
    model_name         : Model label used in the figure title.
    output_path        : File path to save the figure.
    """
    fig, axes = subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, y_true, y_pred, col, title in zip(
        axes,
        [y_true_a, y_true_b],
        [y_pred_a, y_pred_b],
        [_COL_A, _COL_B],
        ["Track A (Imputed)", "Track B (Dropped)"],
    ):
        resid = y_true.values - y_pred
        ax.scatter(y_pred, resid, color=col, alpha=0.4, s=14, edgecolors="none")
        ax.axhline(0, linestyle=":", color=_COL_REF, linewidth=1.5)
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Residuals vs. Predicted — {model_name}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


def plot_feature_importance(model_a, model_b, feat_names_a, feat_names_b, output_path):
    """
    Side-by-side horizontal bar charts of the top 10 feature importances.

    Parameters
    ----------
    model_a, model_b           : Fitted tree estimators with feature_importances_.
    feat_names_a, feat_names_b : Column names from each track's training matrix.
    output_path                : File path to save the figure.
    """
    fig, axes = subplots(1, 2, figsize=(14, 6))

    for ax, model, names, col, title in zip(
        axes,
        [model_a, model_b],
        [feat_names_a, feat_names_b],
        [_COL_A, _COL_B],
        ["Track A (Imputed)", "Track B (Dropped)"],
    ):
        importances = pd.Series(model.feature_importances_, index=names)
        top10 = importances.sort_values(ascending=True).tail(10)

        # Horizontal bars; ascending sort + tail gives longest bar at the top
        ax.barh(top10.index, top10.values, color=col, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Top 10 Feature Importances — DT (Optimal)", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)
