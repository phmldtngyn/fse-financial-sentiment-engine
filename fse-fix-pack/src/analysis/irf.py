"""Local Projections (Jordà 2005) cho impulse response function.

Ưu điểm so với VAR: không cần chỉ định đúng dạng VAR, ước lượng trực tiếp
hồi quy y_{t+h} ~ x_t + controls, h=0..H. Mỗi h là một OLS riêng → dễ diễn giải,
ổn định hơn khi mô hình đúng dạng phi tuyến.

API
---
- ``local_projection_irf(y, x, horizon=14, lags=5, controls=None, alpha=0.05)``
    Trả về DataFrame: horizon, coef, std_err, ci_lo, ci_hi, t, p_value.
- ``lp_irf_table(df, shock_col, target_col, horizon=14, lags=5, group_cols=None)``
    Trả về DataFrame gom nhiều nhóm (theo ticker / feature) với cột group.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _make_lag_block(arr: np.ndarray, lags: int) -> np.ndarray:
    n = len(arr)
    if n <= lags:
        return np.empty((0, lags))
    out = np.full((n, lags), np.nan)
    for k in range(1, lags + 1):
        out[k:, k - 1] = arr[:-k]
    return out


def local_projection_irf(
    y: Sequence[float],
    x: Sequence[float],
    horizon: int = 14,
    lags: int = 5,
    controls: Sequence[Sequence[float]] | None = None,
    alpha: float = 0.05,
    standardize: bool = True,
) -> pd.DataFrame:
    """Ước lượng IRF của y đối với cú sốc x (1 đơn vị, hoặc 1σ nếu standardize).

    Mô hình cho mỗi h:
        y_{t+h} = α_h + β_h · x_t + Σ γ_{h,k} · y_{t-k} + Σ δ_{h,k} · x_{t-k}
                  + controls + ε
    β_h chính là IRF tại horizon h.

    Parameters
    ----------
    horizon : int
        Số kỳ phía trước, mặc định 14.
    lags : int
        Số lag của y và x đưa vào làm control, mặc định 5.
    controls : list of arrays
        Các biến kiểm soát thêm (macro), mỗi array cùng độ dài với y/x.
    standardize : bool
        Nếu True, chuẩn hoá x về đơn vị σ → β_h = phản ứng của y khi x tăng 1σ.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n <= horizon + lags + 5:
        raise ValueError(f"n={n} quá nhỏ cho horizon={horizon}, lags={lags}.")

    if standardize:
        sx = np.std(x)
        if sx > 0:
            x = x / sx

    ctrl_mat = None
    if controls is not None and len(controls) > 0:
        ctrl_mat = np.column_stack([np.asarray(c, dtype=float) for c in controls])
        # lags cũng cho controls
        ctrl_lags = []
        for j in range(ctrl_mat.shape[1]):
            ctrl_lags.append(_make_lag_block(ctrl_mat[:, j], lags))
        ctrl_lag_mat = np.hstack(ctrl_lags) if ctrl_lags else None
    else:
        ctrl_lag_mat = None

    y_lags = _make_lag_block(y, lags)
    x_lags = _make_lag_block(x, lags)

    rows = []
    for h in range(horizon + 1):
        # y_{t+h}: dịch chuyển y lên trước h bước
        if h == 0:
            y_h = y
        else:
            y_h = np.full_like(y, np.nan)
            y_h[:-h] = y[h:]
        # Cắt theo quan sát hợp lệ
        valid = np.isfinite(y_h)
        if ctrl_mat is not None:
            valid &= np.all(np.isfinite(ctrl_mat), axis=1)
        # Build design matrix
        blocks = [np.ones((n, 1)), x.reshape(-1, 1)]
        blocks.append(y_lags)
        blocks.append(x_lags)
        if ctrl_mat is not None:
            blocks.append(ctrl_mat)
        if ctrl_lag_mat is not None:
            blocks.append(ctrl_lag_mat)
        X = np.hstack(blocks)
        valid &= np.all(np.isfinite(X), axis=1)
        Xv = X[valid]
        yv = y_h[valid]
        if len(yv) < Xv.shape[1] + 5:
            rows.append({"horizon": h, "coef": np.nan, "std_err": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan, "t": np.nan,
                         "p_value": np.nan, "n": len(yv)})
            continue
        # OLS
        XtX = Xv.T @ Xv
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ Xv.T @ yv
        resid = yv - Xv @ beta
        dof = len(yv) - Xv.shape[1]
        if dof <= 0:
            continue
        sigma2 = float(resid @ resid) / dof
        var_beta = sigma2 * XtX_inv
        # Hệ số của x là cột index 1 (const=0, x=1)
        idx_x = 1
        se = float(np.sqrt(var_beta[idx_x, idx_x]))
        coef = float(beta[idx_x])
        t = coef / se if se > 0 else float("nan")
        p = float(stats.t.sf(abs(t), dof) * 2) if np.isfinite(t) else float("nan")
        tcrit = float(stats.t.ppf(1 - alpha / 2, dof))
        rows.append({
            "horizon": h, "coef": coef, "std_err": se,
            "ci_lo": coef - tcrit * se, "ci_hi": coef + tcrit * se,
            "t": t, "p_value": p, "n": len(yv),
        })
    return pd.DataFrame(rows)


def lp_irf_table(
    df: pd.DataFrame,
    shock_col: str,
    target_col: str,
    horizon: int = 14,
    lags: int = 5,
    group_cols: Sequence[str] | None = None,
    controls: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Local Projections IRF cho nhiều nhóm."""
    groups = list(group_cols) if group_cols else []
    ctrl_arrays = [df[c].to_numpy() for c in controls] if controls else None
    parts = []
    if groups:
        iterator = df.groupby(groups, dropna=False)
    else:
        iterator = [((), df)]
    for keys, sub in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub = sub.sort_values("date") if "date" in sub.columns else sub
        if controls:
            ctrl_sub = [sub[c].to_numpy() for c in controls]
        else:
            ctrl_sub = None
        try:
            irf = local_projection_irf(
                sub[target_col].to_numpy(), sub[shock_col].to_numpy(),
                horizon=horizon, lags=lags, controls=ctrl_sub, alpha=alpha,
            )
        except ValueError:
            continue
        if irf.empty:
            continue
        irf["shock"] = shock_col
        irf["target"] = target_col
        for k, v in zip(groups, keys):
            irf[k] = v
        parts.append(irf)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
