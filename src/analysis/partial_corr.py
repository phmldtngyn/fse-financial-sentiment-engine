"""Tương quan riêng (partial correlation) kiểm soát macro.

Mục đích: kiểm tra xem sentiment có *sức giải thích gia tăng* (incremental
informativeness) sau khi loại bỏ yếu tố thị trường chung hay không.

Phương pháp: tương quan riêng giữa S_t và target = tương quan Pearson giữa
residual(S_t ~ macro) và residual(target ~ macro).

Cách khác mạnh hơn: regression-based, target ~ S_t + macro, lấy hệ số S_t
và std error, p-value. Module cung cấp cả hai.

API
---
- ``partial_correlation(x, y, controls)``
    Trả về dict {partial_corr, n, n_controls, ...}.
- ``incremental_regression(target, sentiment, controls)``
    Hồi quy target ~ sentiment + controls, trả về sentiment coef, se, t, p.
- ``partial_corr_table(df, s_col, target_col, control_cols, group_cols)``
    Gom cho nhiều nhóm, trả về DataFrame.
- ``incremental_r2_test(target, sentiment, controls)``
    So sánh R² của reduced (target ~ controls) vs full (target ~ sentiment + controls),
    trả về F và p-value.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _resid_against_controls(z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Trả về residual của z sau khi hồi quy trên C (có intercept)."""
    if C.shape[0] == 0:
        return z
    design = np.column_stack([np.ones(len(z)), C])
    beta, *_ = np.linalg.lstsq(design, z, rcond=None)
    return z - design @ beta


def partial_correlation(
    x: Sequence[float],
    y: Sequence[float],
    controls: Sequence[Sequence[float]] | None = None,
) -> dict:
    """Tương quan riêng giữa x và y khi kiểm soát các biến trong controls."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if controls is not None and len(controls) > 0:
        C = np.column_stack([np.asarray(c, dtype=float) for c in controls])
        mask &= np.all(np.isfinite(C), axis=1)
        x, y, C = x[mask], y[mask], C[mask]
    else:
        x, y = x[mask], y[mask]
        C = np.empty((len(x), 0))
    n = len(x)
    k = C.shape[1]
    if n - k - 3 < 1:
        raise ValueError(f"Quá ít bậc tự do: n={n}, k={k}.")
    rx = _resid_against_controls(x, C)
    ry = _resid_against_controls(y, C)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"partial_corr": float("nan"), "n": n, "n_controls": k,
                "p_value": 1.0, "significant": False}
    r = float(np.corrcoef(rx, ry)[0, 1])
    # Fisher z-transform CI
    dof = n - k - 3
    t = r * np.sqrt(dof) / np.sqrt(1 - r * r) if abs(r) < 1 else float("inf")
    p = float(stats.t.sf(abs(t), dof) * 2) if np.isfinite(t) else 0.0
    return {"partial_corr": r, "n": n, "n_controls": k,
            "t_statistic": float(t), "p_value": p,
            "significant": bool(p < 0.05)}


def incremental_regression(
    target: Sequence[float],
    sentiment: Sequence[float],
    controls: Sequence[Sequence[float]] | None = None,
) -> dict:
    """Hồi quy target ~ sentiment + controls. Trả về hệ số sentiment + thống kê."""
    y = np.asarray(target, dtype=float)
    s = np.asarray(sentiment, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    if controls is not None and len(controls) > 0:
        C = np.column_stack([np.asarray(c, dtype=float) for c in controls])
        mask &= np.all(np.isfinite(C), axis=1)
    else:
        C = np.empty((0, 0))
    y, s = y[mask], s[mask]
    if C.shape[1] > 0:
        C = C[mask]
    n = len(y)
    k_full = 1 + C.shape[1] + 1  # intercept + sentiment + controls
    if n - k_full < 5:
        raise ValueError(f"Quá ít bậc tự do: n={n}, k_full={k_full}.")
    design = np.column_stack([np.ones(n), s] + ([C[:, j] for j in range(C.shape[1])] if C.size else []))
    XtX = design.T @ design
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ design.T @ y
    resid = y - design @ beta
    dof = n - k_full
    sigma2 = float(resid @ resid) / dof
    var_beta = sigma2 * XtX_inv
    # sentiment ở vị trí index 1 (const=0, sentiment=1)
    idx_s = 1
    se = float(np.sqrt(var_beta[idx_s, idx_s]))
    coef = float(beta[idx_s])
    t = coef / se if se > 0 else float("nan")
    p = float(stats.t.sf(abs(t), dof) * 2) if np.isfinite(t) else float("nan")
    tcrit = float(stats.t.ppf(0.975, dof))
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    ssr = float(resid @ resid)
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    return {
        "sentiment_coef": coef, "sentiment_se": se,
        "sentiment_t": t, "sentiment_pvalue": p,
        "sentiment_ci_lo": coef - tcrit * se,
        "sentiment_ci_hi": coef + tcrit * se,
        "sentiment_significant": bool(np.isfinite(p) and p < 0.05),
        "r2_full": r2, "n": n, "dof": dof, "n_controls": C.shape[1],
    }


def incremental_r2_test(
    target: Sequence[float],
    sentiment: Sequence[float],
    controls: Sequence[Sequence[float]] | None = None,
) -> dict:
    """F-test so sánh R² của reduced vs full. Trả lời: sentiment có tăng R² không?"""
    y = np.asarray(target, dtype=float)
    s = np.asarray(sentiment, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    if controls is not None and len(controls) > 0:
        C = np.column_stack([np.asarray(c, dtype=float) for c in controls])
        mask &= np.all(np.isfinite(C), axis=1)
    else:
        C = np.empty((0, 0))
    y, s = y[mask], s[mask]
    if C.shape[1] > 0:
        C = C[mask]
    n = len(y)
    if n < 10:
        raise ValueError(f"n={n} quá nhỏ.")
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    # Reduced: y ~ const + controls
    if C.shape[1] > 0:
        Xr = np.column_stack([np.ones(n), C])
    else:
        Xr = np.ones((n, 1))
    beta_r, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    rss_r = float(np.sum((y - Xr @ beta_r) ** 2))
    # Full: y ~ const + sentiment + controls
    Xf = np.column_stack([np.ones(n), s] + ([C[:, j] for j in range(C.shape[1])] if C.size else []))
    beta_f, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    rss_f = float(np.sum((y - Xf @ beta_f) ** 2))
    k_r = Xr.shape[1]
    k_f = Xf.shape[1]
    dof_num = k_f - k_r
    dof_den = n - k_f
    if dof_den <= 0 or rss_f <= 0:
        return {"F": float("nan"), "p_value": 1.0, "significant": False,
                "r2_reduced": float("nan"), "r2_full": float("nan"),
                "delta_r2": float("nan"), "n": n}
    F = ((rss_r - rss_f) / dof_num) / (rss_f / dof_den)
    p = float(stats.f.sf(F, dof_num, dof_den))
    r2_r = 1 - rss_r / sst if sst > 0 else float("nan")
    r2_f = 1 - rss_f / sst if sst > 0 else float("nan")
    return {
        "F": float(F), "p_value": p,
        "significant": bool(p < 0.05),
        "r2_reduced": r2_r, "r2_full": r2_f,
        "delta_r2": float(r2_f - r2_r), "n": n,
        "dof_num": dof_num, "dof_den": dof_den,
    }


def partial_corr_table(
    df: pd.DataFrame,
    s_col: str,
    target_col: str,
    control_cols: Sequence[str],
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Partial correlation + incremental R² cho nhiều nhóm."""
    groups = list(group_cols) if group_cols else []
    parts = []
    if groups:
        iterator = df.groupby(groups, dropna=False)
    else:
        iterator = [((), df)]
    for keys, sub in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub = sub.sort_values("date") if "date" in sub.columns else sub
        ctrl = [sub[c].to_numpy() for c in control_cols] if control_cols else None
        try:
            pc = partial_correlation(sub[s_col].to_numpy(), sub[target_col].to_numpy(), ctrl)
            ir = incremental_regression(sub[target_col].to_numpy(), sub[s_col].to_numpy(), ctrl)
            rt = incremental_r2_test(sub[target_col].to_numpy(), sub[s_col].to_numpy(), ctrl)
        except ValueError as e:
            row = dict(zip(groups, keys))
            row.update({"error": str(e), "partial_corr": float("nan")})
            parts.append(row)
            continue
        row = dict(zip(groups, keys))
        row.update({
            "s_col": s_col, "t_col": target_col,
            "partial_corr": pc["partial_corr"], "pc_pvalue": pc["p_value"],
            "pc_significant": pc["significant"],
            "sentiment_coef": ir["sentiment_coef"],
            "sentiment_se": ir["sentiment_se"],
            "sentiment_pvalue": ir["sentiment_pvalue"],
            "sentiment_significant": ir["sentiment_significant"],
            "r2_reduced": rt["r2_reduced"], "r2_full": rt["r2_full"],
            "delta_r2": rt["delta_r2"], "F_test": rt["F"],
            "F_pvalue": rt["p_value"], "F_significant": rt["significant"],
            "n": ir["n"], "n_controls": ir["n_controls"],
        })
        parts.append(row)
    return pd.DataFrame(parts)
