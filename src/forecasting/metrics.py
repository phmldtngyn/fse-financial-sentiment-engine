"""Metrics cho forecasting: RMSE, MAE, MAPE, R², Diebold-Mariano."""
import numpy as np
from scipy import stats


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    denom = np.abs(y_true)
    mask = denom > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask])) * 100)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def all_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "n": len(y_true),
    }


def diebold_mariano(errors1, errors2, h=1, power=2):
    """DM test: có sự khác biệt ý nghĩa giữa 2 mô hình không.

    Returns (stat, p_value). p < 0.05 → khác biệt ý nghĩa.
    """
    d = np.abs(errors1) ** power - np.abs(errors2) ** power
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / n
    if var_d <= 0:
        return float("nan"), float("nan")
    stat = mean_d / np.sqrt(var_d)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)