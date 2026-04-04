"""
    Model training, cross-validation tuning, and evaluation functions for 
    the null model, linear regression, Lasso, decision tree regressor, and 
    random forest regressor.
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from src.config import CV_FOLDS, RANDOM_STATE

def tune_decision_tree(X_train, y_train, depths):
    """
    Finds the optimal tree depth via 10-fold CV using the one-standard-error rule.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix.
    y_train : pandas.Series or numpy.ndarray
        Training target vector.
    depths : list of int
        Candidate tree depths to evaluate.

    Returns
    -------
    best_depth : int
        Shallowest depth within 1 SE of the minimum CV RMSE.
    history : list of dict
        Tuning metrics per depth (depth, rmse, std).
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    history = []

    for d in depths:
        scores = cross_val_score(
            DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE),
            X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error',
            n_jobs=-1
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
    Finds the optimal number of trees via 10-fold CV using the one-standard-error rule.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix.
    y_train : pandas.Series or numpy.ndarray
        Training target vector.
    n_estimators_list : list of int
        Candidate forest sizes to evaluate.

    Returns
    -------
    best_n : int
        Fewest trees within 1 SE of the minimum CV RMSE.
    history : list of dict
        Tuning metrics per forest size (n_estimators, rmse, std).
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
    Finds the optimal Lasso alpha via 10-fold CV using the one-standard-error rule.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix.
    y_train : pandas.Series or numpy.ndarray
        Training target vector.
    alphas : array-like
        Candidate regularization strengths to evaluate.

    Returns
    -------
    best_alpha : float
        Largest (most regularized) alpha within 1 SE of the minimum CV RMSE.
    history : list of dict
        Tuning metrics per alpha (alpha, rmse, std).
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    history = []

    for alpha in alphas:
        scores = cross_val_score(
            Lasso(alpha=alpha, max_iter=10_000),
            X_train, y_train,
            cv=kf, scoring='neg_root_mean_squared_error',
            n_jobs=-1
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
    Fits an estimator on the training set and evaluates it on the hold-out test set.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame or numpy.ndarray
        Training and testing feature matrices.
    y_train, y_test : pandas.Series or numpy.ndarray
        Training and testing target vectors.
    model : estimator object
        Unfitted Scikit-Learn estimator.

    Returns
    -------
    metrics : dict
        Test set performance metrics (RMSE, R²).
    y_pred : numpy.ndarray
        Model predictions on the test set.
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
    Computes the 10-fold CV RMSE for a baseline Linear Regression model.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix.
    y_train : pandas.Series or numpy.ndarray
        Training target vector.

    Returns
    -------
    mean_rmse : float
        Mean cross-validated root mean squared error.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        LinearRegression(), X_train, y_train,
        cv=kf, scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    return float(-np.mean(scores))
