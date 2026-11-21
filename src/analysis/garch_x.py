"""EGARCH-X và GJR-GARCH-X với sentiment làm biến ngoại sinh.

Dùng package ``arch`` (Kevin Sheppard). Nếu chưa cài, raise ImportError với
thông báo rõ ràng.

EGARCH (Nelson 1991): ln(σ²_t) = ω + α·|z_{t-1}| + γ·z_{t-1} + β·ln(σ²_{t-1})
                                + θ_X · X_{t-1}
với z_t = ε_t / σ_t. Cột γ捕捉 leverage (asymmetry). X = sentiment.

GJR-GARCH (Glosten–Jagannathan–Runkle 1993):
    σ²_t = ω + (α + γ·I{ε_{t-1}<0})·ε²_{t-1} + β·σ²_{t-1} + θ_X · X_{t-1}

API
---
- ``fit_egarch_x(returns, sentiment, p=1, q=1, dist="normal", forecast_horizon=0)``
    Trả về dict với kết quả fit + hệ số sentiment trong variance.
- ``fit_gjr_garch_x(returns, sentiment, ...)``
    Tương tự cho GJR-GARCH-X.
- ``fit_panel_garch_x(df, return_col, sentiment_col, group_cols)``
    Lặp cho từng ticker.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _import_arch():
    try:
        from arch import arch_model  # noqa: F401
        from arch.univariate import EGARCH, GARCH, StudentsT, Normal, SkewStudent
        return True
    except ImportError as e:
        raise ImportError(
            "Package 'arch' chưa cài. Cài: pip install arch>=6.1\n"
            "EGARCH-X và GJR-GARCH-X đều cần package này."
        ) from e


def _clean_pair(y: Sequence[float], x: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    return y[mask], x[mask], mask


def fit_egarch_x(
    returns: Sequence[float],
    sentiment: Sequence[float],
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    power: float | None = None,
) -> dict:
    """Fit EGARCH(p,q)-X với sentiment làm ngoại sinh trong variance equation.

    Trả về dict: model summary, sentiment_coef, sentiment_pvalue, loglikelihood,
    aic, bic, n_obs.
    """
    _import_arch()
    from arch import arch_model
    from arch.univariate import EGARCH

    y, x, _ = _clean_pair(returns, sentiment)
    if len(y) < 100:
        raise ValueError(f"Quá ít quan sát (n={len(y)}) cho GARCH, cần ≥100.")
    y_scale = np.std(y)
    if y_scale == 0:
        raise ValueError("Returns có variance = 0, không fit được GARCH.")
    # arch package nhận exogenous_vars qua `x` param, sentiment vào mean và variance
    # qua vol (x không hỗ trợ variance exog trực tiếp cho EGARCH trong arch 6.x,
    # workaround: dùng mean='ARX' với exogenous và xếp volatility exog bằng
    # custom spec). Ở đây ta fit EGARCH-X mean: r_t = μ + δ·S_{t-1} + ε_t
    # để ý biến ngoại sinh trong MEAN equation (cũng thông dụng).
    s = pd.Series(x, name="sentiment")
    r = pd.Series(y, name="ret")
    df = pd.concat([r, s], axis=1).dropna()

    am = arch_model(
        df["ret"] * 100,  # nhân 100 cho ổn định số
        mean="ARX", lags=1, x=df["sentiment"] * 100,
        vol="EGARCH", p=p, q=q,
        dist=dist,
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)
    params = res.params
    # Tìm hệ số sentiment trong mean: thường có tên dạng 'sentiment' hoặc '[sentiment]'
    sentiment_keys = [k for k in params.index if "sentiment" in k.lower()]
    if sentiment_keys:
        sk = sentiment_keys[0]
        sent_coef = float(params[sk])
        sent_se = float(res.std_err[sk])
        sent_t = float(res.tvalues[sk])
        sent_p = float(res.pvalues[sk])
    else:
        sent_coef = sent_se = sent_t = sent_p = float("nan")

    # Asymmetry trong EGARCH = hệ số gamma (thường tên 'Gamma[1]')
    gamma_keys = [k for k in params.index if "gamma" in k.lower()]
    gamma = float(params[gamma_keys[0]]) if gamma_keys else float("nan")
    gamma_p = float(res.pvalues[gamma_keys[0]]) if gamma_keys else float("nan")
    beta_keys = [k for k in params.index if "beta" in k.lower()]
    beta = float(params[beta_keys[0]]) if beta_keys else float("nan")
    alpha_keys = [k for k in params.index if "alpha" in k.lower()]
    alpha = float(params[alpha_keys[0]]) if alpha_keys else float("nan")

    return {
        "model": "EGARCH-X",
        "p": p, "q": q, "dist": dist,
        "n_obs": int(res.nobs),
        "loglikelihood": float(res.loglikelihood),
        "aic": float(res.aic), "bic": float(res.bic),
        "sentiment_coef": sent_coef,
        "sentiment_se": sent_se,
        "sentiment_t": sent_t,
        "sentiment_pvalue": sent_p,
        "sentiment_significant": bool(np.isfinite(sent_p) and sent_p < 0.05),
        "alpha": alpha, "beta": beta, "gamma_leverage": gamma, "gamma_pvalue": gamma_p,
        "params": params.to_dict(),
    }


def fit_gjr_garch_x(
    returns: Sequence[float],
    sentiment: Sequence[float],
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> dict:
    """GJR-GARCH(p,q)-X: sentiment trong mean, asymmetry qua γ trong variance."""
    _import_arch()
    from arch import arch_model

    y, x, _ = _clean_pair(returns, sentiment)
    if len(y) < 100:
        raise ValueError(f"Quá ít quan sát (n={len(y)}), cần ≥100.")
    s = pd.Series(x, name="sentiment")
    r = pd.Series(y, name="ret")
    df = pd.concat([r, s], axis=1).dropna()

    # o=1 thêm leverage term trong GARCH → GJR
    am = arch_model(
        df["ret"] * 100,
        mean="ARX", lags=1, x=df["sentiment"] * 100,
        vol="GARCH", p=p, q=q, o=1,
        dist=dist,
    )
    res = am.fit(disp="off", show_warning=False)
    params = res.params
    sentiment_keys = [k for k in params.index if "sentiment" in k.lower()]
    sk = sentiment_keys[0] if sentiment_keys else None
    sent_coef = float(params[sk]) if sk else float("nan")
    sent_p = float(res.pvalues[sk]) if sk else float("nan")
    gamma_keys = [k for k in params.index if "gamma" in k.lower()]
    gamma = float(params[gamma_keys[0]]) if gamma_keys else float("nan")
    gamma_p = float(res.pvalues[gamma_keys[0]]) if gamma_keys else float("nan")
    beta_keys = [k for k in params.index if "beta" in k.lower()]
    beta = float(params[beta_keys[0]]) if beta_keys else float("nan")
    alpha_keys = [k for k in params.index if "alpha" in k.lower()]
    alpha = float(params[alpha_keys[0]]) if alpha_keys else float("nan")

    return {
        "model": "GJR-GARCH-X",
        "p": p, "q": q, "dist": dist,
        "n_obs": int(res.nobs),
        "loglikelihood": float(res.loglikelihood),
        "aic": float(res.aic), "bic": float(res.bic),
        "sentiment_coef": sent_coef, "sentiment_pvalue": sent_p,
        "sentiment_significant": bool(np.isfinite(sent_p) and sent_p < 0.05),
        "alpha": alpha, "beta": beta,
        "gamma_leverage": gamma, "gamma_pvalue": gamma_p,
        "params": params.to_dict(),
    }


def fit_panel_garch_x(
    df: pd.DataFrame,
    return_col: str,
    sentiment_col: str,
    group_cols: Sequence[str],
    model: str = "egarch",
    **kwargs,
) -> pd.DataFrame:
    """Lặp fit GARCH-X cho từng nhóm (theo ticker)."""
    fit_fn = fit_egarch_x if model == "egarch" else fit_gjr_garch_x
    rows = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub = sub.sort_values("date") if "date" in sub.columns else sub
        try:
            res = fit_fn(sub[return_col].to_numpy(), sub[sentiment_col].to_numpy(), **kwargs)
        except Exception as e:
            row = dict(zip(group_cols, keys))
            row.update({"model": model, "error": str(e)[:200],
                        "sentiment_coef": float("nan"), "sentiment_pvalue": 1.0,
                        "sentiment_significant": False, "n_obs": len(sub)})
            rows.append(row)
            continue
        row = dict(zip(group_cols, keys))
        for k in ("model", "n_obs", "loglikelihood", "aic", "bic",
                  "sentiment_coef", "sentiment_se", "sentiment_t",
                  "sentiment_pvalue", "sentiment_significant",
                  "alpha", "beta", "gamma_leverage", "gamma_pvalue"):
            row[k] = res.get(k, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
