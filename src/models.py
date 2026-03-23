"""
    Factory functions for the null model, linear regression,
    and decision tree regressor.
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from src.config import CV_FOLDS, RANDOM_STATE

def tune_decision_tree(X_train, y_train, depths):
    """
    Finds the optimal tree depth using 10-fold CV on the 80% Training Set only.

    Parameters
    ----------
    X_train : Training feature matrix (80%).
    y_train : Training response vector.
    depths  : List of max_depth integers to test.

    Returns
    -------
    best_depth : Integer depth with the lowest mean RMSE.
    history    : List of dictionaries for plotting the tuning curve.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    history = []
    
    for d in depths:
        model = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)
        # CV only on training data (80% portion)
        scores = cross_val_score(
            model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error'
        )
        avg_rmse = -np.mean(scores)
        std_rmse = np.std(-scores)   # needed for ±1 s.d. band in tuning plot
        history.append({"depth": d, "rmse": avg_rmse, "std": std_rmse})
            
    # Identify depth with the minimum mean RMSE
    best_depth = min(history, key=lambda x: x['rmse'])['depth']
    return best_depth, history

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
