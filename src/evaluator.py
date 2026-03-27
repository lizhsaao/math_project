"""
    Formats the final results into a comparison table.
"""
import pandas as pd


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
