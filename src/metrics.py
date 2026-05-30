"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def scaled_r2(y_true, y_pred) -> float:
    """Competition score: max(0, 100 * R²)."""
    r2 = r2_score(y_true, y_pred)
    return max(0.0, 100.0 * r2)


def regression_metrics(y_true, y_pred) -> dict:
    r2 = r2_score(y_true, y_pred)
    return {
        "r2": r2,
        "scaled_r2": max(0.0, 100.0 * r2),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
