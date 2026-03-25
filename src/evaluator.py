"""
    Formats the final results into a comparison table. 
"""
import pandas as pd

def create_summary_table(results_a, results_b):
    """
    Formats the results of both tracks into a clean comparison.
    """
    data = []
    # Match the five models you requested
    models = ["Null Model", "Linear Regression", "DT (Depth 3)", "DT (Unconstrained)", "DT (Optimal)", "Random Forest"]
    
    for m in models:
        data.append({
            "Model": m,
            "Track A (RMSE)": f"{results_a[m]['rmse']:.4f}",
            "Track A R²": f"{results_a[m]['r2']:.4f}",
            "Track B (RMSE)": f"{results_b[m]['rmse']:.4f}",
            "Track B R²": f"{results_b[m]['r2']:.4f}"
        })
    return pd.DataFrame(data)