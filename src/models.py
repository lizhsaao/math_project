"""
    Factory functions for the null model, linear regression, Lasso,
    decision tree regressor, and random forest regressor.
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from src.config import CV_FOLDS, RANDOM_STATE

def tune_decision_tree(X_train, y_train, depths):
    """
    Finds the optimal tree depth using 10-fold CV on the 80% training set only,
    applying the one-standard-error rule: select the shallowest depth whose
    mean CV RMSE lies within one SE of the minimum.

    Parameters
    ----------
    X_train : Training feature matrix (80%).
    y_train : Training response vector.
    depths  : List of max_depth integers to test.

    Returns
    -------
    best_depth : Shallowest depth within 1 SE of the minimum CV RMSE.
    history    : List of {"depth", "rmse", "std"} dicts for plotting.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    history = []

    for d in depths:
        scores = cross_val_score(
            DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE),
            X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error'
        )
        history.append({"depth": d, "rmse": float(-np.mean(scores)),
                        "std": float(np.std(-scores))})

    # 1-SE rule: shallowest depth whose lower ±1 s.d. bound touches the minimum RMSE.
    # Equivalent to finding where (rmse - std) ≤ min_rmse on the tuning curve.
    rmse_arr  = np.array([h['rmse'] for h in history])
    std_arr   = np.array([h['std']  for h in history])
    min_rmse  = float(rmse_arr[int(np.argmin(rmse_arr))])
    lower_arr = rmse_arr - std_arr
    best_depth = next(h['depth'] for h, lb in zip(history, lower_arr) if lb <= min_rmse)
    return best_depth, history

def tune_random_forest(X_train, y_train, n_estimators_list):
    """
    Scans n_estimators via 10-fold CV on the 80% training set only, applying
    the one-standard-error rule: select the fewest trees whose mean CV RMSE
    lies within one SE of the minimum.

    Parameters
    ----------
    X_train          : Training feature matrix (80%).
    y_train          : Training response vector.
    n_estimators_list: List of n_estimators values to evaluate.

    Returns
    -------
    best_n  : Fewest trees within 1 SE of the minimum CV RMSE.
    history : List of {"n_estimators", "rmse", "std"} dicts (one per candidate).
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    history = []

    for n in n_estimators_list:
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1),
            X_train, y_train,
            cv=kf, scoring='neg_root_mean_squared_error'
        )
        history.append({
            "n_estimators": n,
            "rmse": float(-np.mean(scores)),
            "std":  float(np.std(-scores))
        })

    # 1-SE rule: fewest trees whose lower ±1 s.d. bound touches the minimum RMSE.
    rmse_arr  = np.array([h['rmse'] for h in history])
    std_arr   = np.array([h['std']  for h in history])
    min_rmse  = float(rmse_arr[int(np.argmin(rmse_arr))])
    lower_arr = rmse_arr - std_arr
    best_n = next(h['n_estimators'] for h, lb in zip(history, lower_arr) if lb <= min_rmse)
    return best_n, history


def tune_lasso(X_train, y_train, alphas):
    """
    Finds the optimal Lasso regularisation strength (alpha) via 10-fold CV on
    the 80% training set only, applying the one-standard-error rule: select the
    LARGEST alpha (most regularised / most parsimonious model) whose lower ±1 s.d.
    bound still lies within one SE of the minimum CV RMSE.

    Parameters
    ----------
    X_train : Training feature matrix (80%).
    y_train : Training response vector.
    alphas  : Sequence of positive alpha values to evaluate (e.g. log-spaced).

    Returns
    -------
    best_alpha : Largest alpha within 1 SE of the minimum CV RMSE.
    history    : List of {"alpha", "rmse", "std"} dicts (one per candidate).
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    history = []

    for alpha in alphas:
        scores = cross_val_score(
            Lasso(alpha=alpha, max_iter=10_000),
            X_train, y_train,
            cv=kf, scoring='neg_root_mean_squared_error'
        )
        history.append({
            "alpha": float(alpha),
            "rmse":  float(-np.mean(scores)),
            "std":   float(np.std(-scores)),
        })

    # 1-SE rule for Lasso: prefer the LARGEST alpha (most regularised) whose
    # lower ±1 s.d. bound touches the minimum RMSE. Scan history in reverse so
    # the first qualifying match is the largest qualifying alpha.
    rmse_arr  = np.array([h['rmse'] for h in history])
    std_arr   = np.array([h['std']  for h in history])
    min_rmse  = float(rmse_arr[int(np.argmin(rmse_arr))])
    lower_arr = rmse_arr - std_arr
    best_alpha = next(
        h['alpha']
        for h, lb in zip(reversed(history), lower_arr[::-1])
        if lb <= min_rmse
    )
    return best_alpha, history


def get_metrics_and_preds(X_train, y_train, X_test, y_test, model):
    """
    Trains on the 80% set and evaluates on the 'locked' 20% test set.

    Parameters
    ----------
    X_train, y_train : Data used to fit the model.
    X_test, y_test   : Held-out data used for final performance check.
    model            : Initialised Scikit-Learn estimator.

    Returns
    -------
    metrics : Dictionary with RMSE and R2 scores.
    y_pred  : Array of predictions for actual-vs-predicted plots.
    """
    # 1. Fit on the training set only
    model.fit(X_train, y_train)
    
    # 2. Predict on the 'locked' test set (20% portion)
    y_pred = model.predict(X_test)
    
    # 3. Calculate final metrics for the report
    metrics = {
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    return metrics, y_pred

def cv_rmse_lr(X_train, y_train):
    """
    Computes the 10-fold CV RMSE for Linear Regression on the training set.
    Used as the reference baseline on the DT tuning curve, so both the curve
    and the baseline are on the same metric (CV RMSE, not hold-out RMSE).

    Parameters
    ----------
    X_train : Training feature matrix (80%).
    y_train : Training response vector.

    Returns
    -------
    mean_rmse : Mean CV RMSE across folds.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        LinearRegression(), X_train, y_train,
        cv=kf, scoring='neg_root_mean_squared_error'
    )
    return float(-np.mean(scores))
