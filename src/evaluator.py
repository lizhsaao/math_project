"""
    Formats the final results into a comparison table.
"""
import pandas as pd


def create_summary_table(results_a, results_b, models=None):
    """
    Formats model evaluation results for both tracks into a comparison table.

    Parameters
    ----------
    results_a, results_b : Dicts mapping model name -> {"rmse", "r2"} for each track.
    models               : Which models to include in the table.
                           - None (default): all models present in results_a,
                             in insertion order.
                           - str: a single model name, e.g. "Lasso".
                           - list[str]: an explicit ordered subset of model names.

    Returns
    -------
    pd.DataFrame with columns for model name, RMSE, and R² on both tracks.
    """
    if models is None:
        model_list = list(results_a.keys())
    elif isinstance(models, str):
        model_list = [models]
    else:
        model_list = list(models)

    data = []
    for m in model_list:
        data.append({
            "Model":           m,
            "Track A (RMSE)":  f"{results_a[m]['rmse']:.4f}",
            "Track A R²":      f"{results_a[m]['r2']:.4f}",
            "Track B (RMSE)":  f"{results_b[m]['rmse']:.4f}",
            "Track B R²":      f"{results_b[m]['r2']:.4f}",
        })
    return pd.DataFrame(data)
