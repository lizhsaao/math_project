import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error


def run_cv(model, X, y, cv_folds, random_state):
    """
    10-fold (or cv_folds-fold) cross-validation using neg_root_mean_squared_error.

    Returns (mean_rmse, std_rmse) where std_rmse is the standard deviation of
    per-fold RMSE scores (not the standard error).
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    result = cross_validate(
        model, X, y, cv=cv, scoring="neg_root_mean_squared_error"
    )
    scores = -result["test_score"]
    return float(scores.mean()), float(scores.std())


def one_se_rule(rmse_list, std_list):
    """
    One-standard-error rule for selecting the simplest model within one SE of
    the minimum CV RMSE.

    Given parallel lists of CV RMSE and per-fold SD (one entry per depth),
    returns the index of the shallowest depth whose RMSE falls at or below
    rmse_min + std_at_min.
    """
    rmse = np.asarray(rmse_list)
    std  = np.asarray(std_list)
    i_min = int(np.argmin(rmse))
    threshold = rmse[i_min] + std[i_min]
    # First index (simplest model) whose RMSE is within one SE of the minimum
    for i, r in enumerate(rmse):
        if r <= threshold:
            return i
    return i_min  # fallback: return the argmin itself


def holdout_rmse(model, X_train, y_train, X_test, y_test):
    """
    Fit model on (X_train, y_train) and return RMSE on (X_test, y_test).
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return float(np.sqrt(mean_squared_error(y_test, predictions)))
