"""Kiểm định nhân quả Granger và Toda–Yamamoto.

Hai phép thử nhân quả:

1. **Granger (1969)** — kiểm định Wald cho tính tất định của x đối với y trong
   VAR bivariate. Yêu cầu chuỗi dừng (stationary). Trả về F-statistic và p-value.

2. **Toda–Yamamoto (1995)** — version robust với unit root/cointegration. Ước
   lượng VAR(p+d) với d = bậc tích hợp tối đa, nhưng Wald test chỉ trên p hệ số
   đầu tiên. Khuyến nghị dùng khi chuỗi tài chính thường có unit root.

API
---
- ``granger_test(y, x, max_lag=10, alpha=0.05)``
    Trả về DataFrame các (lag, F, p_value, significant) — chọn lag tốt nhất theo
    BIC và chạy Wald cho tất cả lag 1..max_lag.
- ``toda_yamamoto_test(y, x, max_lag=10, d=1, alpha=0.05)``
    Tương tự nhưng thêm d lags dummy.
- ``bidirectional_causality(df, s_col, target_col, max_lag=10, ...)``
    Chạy cả 2 hướng: S→target và target→S. Trả về DataFrame rõ ràng.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _make_lagged_matrix(series: Sequence[float], max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (Y_target, X_lagged, lag_matrix) cho hồi quy y_t trên các lag của x và y."""
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n <= max_lag + 2:
        raise ValueError(f"n={n} quá nhỏ cho max_lag={max_lag}.")
    rows = []
    y_arr = []
    for t in range(max_lag, n):
        row = [s[t - k] for k in range(1, max_lag + 1)]
        rows.append(row)
        y_arr.append(s[t])
    X = np.array(rows, dtype=float)
    Y = np.array(y_arr, dtype=float)
    return Y, X, np.arange(1, max_lag + 1)


def _wald_restricted(Y: np.ndarray, X_full: np.ndarray, k_restricted: int) -> tuple[float, float, int, int]:
    """Wald test cho H0: các hệ số của k_restricted biến đầu = 0.

    Mô hình unrestricted: Y = X_full @ beta + eps
    Mô hình restricted  : Y = X_full[:, k_restricted:] @ gamma + eps
    """
    n, k_full = X_full.shape
    if n <= k_full + 5:
        return float("nan"), 1.0, k_restricted, n - k_full - 1
    beta_u, *_ = np.linalg.lstsq(X_full, Y, rcond=None)
    resid_u = Y - X_full @ beta_u
    rss_u = float(resid_u @ resid_u)
    X_r = X_full[:, k_restricted:]
    beta_r, *_ = np.linalg.lstsq(X_r, Y, rcond=None)
    resid_r = Y - X_r @ beta_r
    rss_r = float(resid_r @ resid_r)
    if rss_u <= 0:
        return float("nan"), 1.0, k_restricted, n - k_full - 1
    df_num = k_restricted
    df_den = n - k_full
    F = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    p = float(stats.f.sf(F, df_num, df_den))
    return float(F), p, df_num, df_den


def granger_test(
    y: Sequence[float],
    x: Sequence[float],
    max_lag: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Granger causality x → y.

    Mô hình unrestricted: y_t = const + Σ a_i y_{t-i} + Σ b_j x_{t-j} + eps
    H0: b_1 = ... = b_p = 0  (x không Granger-cause y).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    rows = []
    for p in range(1, max_lag + 1):
        if n <= 2 * p + 5:
            continue
        Y, X_y, _ = _make_lagged_matrix(y, p)
        _, X_x, _ = _make_lagged_matrix(x, p)
        # Cắt cho khớp độ dài (đã khớp theo construction)
        const = np.ones((Y.shape[0], 1))
        X_full = np.column_stack([const, X_y, X_x])  # const + p lags of y + p lags of x
        # H0: hệ số của X_x = 0  → số biến restricted = p
        F, pval, df_n, df_d = _wald_restricted(Y, X_full, p)
        rows.append({
            "lag": p, "F_statistic": F, "p_value": pval,
            "df_num": df_n, "df_den": df_d,
            "significant": bool(pval < alpha),
        })
    return pd.DataFrame(rows)


def toda_yamamoto_test(
    y: Sequence[float],
    x: Sequence[float],
    max_lag: int = 10,
    d: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Toda–Yamamoto (1995) causality, robust với unit root/cointegration.

    Ước lượng VAR(p+d) nhưng Wald test chỉ trên p hệ số đầu của x. Mặc định d=1
    (giả định I(1) tối đa). Nếu cả 2 chuỗi I(0), set d=0 → trở về Granger.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    p_eff = max_lag + d
    n = len(y)
    rows = []
    for p in range(1, max_lag + 1):
        if n <= 2 * p_eff + 5:
            continue
        Y, X_y, _ = _make_lagged_matrix(y, p_eff)
        _, X_x, _ = _make_lagged_matrix(x, p_eff)
        const = np.ones((Y.shape[0], 1))
        X_full = np.column_stack([const, X_y, X_x])
        # Wald chỉ trên p hệ số đầu của x (vị trí const+p..const+p+p-1 trong X_full?
        # Thứ tự cột: const (1) + X_y (p_eff) + X_x (p_eff)
        # Hệ số x_{t-1..t-p} tương ứng với p cột đầu của X_x.
        # Trong _wald_restricted, "k_restricted" = số biến ở ĐẦU bị zero — cần đảo ngược.
        # Xây lại: đặt X_full = [const, X_x[:, :p], X_y, X_x[:, p:]]
        X_full2 = np.column_stack([const, X_x[:, :p], X_y, X_x[:, p:]])
        F, pval, df_n, df_d = _wald_restricted(Y, X_full2, p)
        rows.append({
            "lag": p, "F_statistic": F, "p_value": pval,
            "df_num": df_n, "df_den": df_d, "d_extra": d,
            "significant": bool(pval < alpha),
        })
    return pd.DataFrame(rows)


def bidirectional_causality(
    df: pd.DataFrame,
    s_col: str,
    target_col: str,
    max_lag: int = 10,
    method: str = "granger",
    d: int = 1,
    alpha: float = 0.05,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Chạy cả 2 hướng S→target và target→S.

    Trả về DataFrame với cột: direction, lag, F, p_value, significant (+group).
    """
    test_fn = granger_test if method == "granger" else (
        lambda yy, xx: toda_yamamoto_test(yy, xx, max_lag=max_lag, d=d, alpha=alpha)
    )
    parts = []
    groups = list(group_cols) if group_cols else []
    if groups:
        iterator = df.groupby(groups, dropna=False)
    else:
        iterator = [((), df)]
    for keys, sub in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub = sub.sort_values("date") if "date" in sub.columns else sub
        s = sub[s_col].to_numpy()
        t = sub[target_col].to_numpy()
        if len(s) < 2 * max_lag + 10:
            continue
        # S → target
        res_s2t = test_fn(t, s)
        if not res_s2t.empty:
            res_s2t["direction"] = f"{s_col}->{target_col}"
            res_s2t["method"] = method
            for k, v in zip(groups, keys):
                res_s2t[k] = v
            parts.append(res_s2t)
        # target → S
        res_t2s = test_fn(s, t)
        if not res_t2s.empty:
            res_t2s["direction"] = f"{target_col}->{s_col}"
            res_t2s["method"] = method
            for k, v in zip(groups, keys):
                res_t2s[k] = v
            parts.append(res_t2s)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
