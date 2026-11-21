"""Kiểm định điểm gãy cấu trúc cho chuỗi tương quan.

Hai phép thử bổ sung cho bảng tương quan:

1. **Bai–Perron (1998, 2003)** multiple breakpoints — ước lượng các điểm gãy
   trong quan hệ hồi quy y = α + β·x + ε, tối đa K điểm, dùng dynamic algorithm.
   Phụ thuộc ``statsmodels.regression.linear_model.OLS`` và tìm grid tối ưu.

2. **CUSUM of squares (Brown, Durbin, Evans 1975)** — phát hiện thay đổi cấu
   trúc tổng quát trong chuỗi hồi quy, không cần biết trước số điểm gãy.

API
---
- ``bai_perron_breaks(y, x, max_breaks=3, min_seg=60, alpha=0.05)``
    Trả về dict {n_breaks: int, break_dates: list[int], supF: float, ...}.
- ``cusum_squares(y, x, alpha=0.05)``
    Trả về dict {statistic, p_value, significant, max_pos}.
- ``rolling_corr_stability(df, s_col, t_col, group_cols, window=120)``
    Trả về DataFrame rolling correlation theo từng nhóm để visual stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _rss(y: np.ndarray, X: np.ndarray) -> float:
    """Residual sum of squares của OLS."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def _design(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x])


@dataclass
class BaiPerronResult:
    n_breaks: int
    break_indices: list[int] = field(default_factory=list)
    sup_f: float = float("nan")
    p_value: float = float("nan")
    significant: bool = False
    n_obs: int = 0
    method: str = "bai_perron_grid"

    def as_dict(self) -> dict:
        return self.__dict__


def bai_perron_breaks(
    y: Sequence[float],
    x: Sequence[float],
    max_breaks: int = 3,
    min_seg: int = 60,
    alpha: float = 0.05,
    crit_value: float | None = None,
) -> BaiPerronResult:
    """Tìm tối đa ``max_breaks`` điểm gãy trong hồi quy y ~ x.

    Đây là phiên bản dynamic-programming đơn giản hoá:
    - Grid các vị trí ứng viên = mọi m trong [min_seg, n-min_seg].
    - Với mỗi số điểm gãy k ∈ {1,...,max_breaks}, tìm cấu hình minimise tổng RSS.
    - Chọn k bằng supF test ( Andrews 1993 / Bai–Perron 2003 ) với ngưỡng kinh
      điển cho alpha=0.05 (Bai 1997, Table 1): ≈ 11.47 cho 1 break với p=2.
      Nếu ``crit_value``=None, dùng các ngưỡng 近似 cho p=2 (hệ số + hằng).

    Tham số
    -------
    min_seg : int
        Độ dài tối thiểu một phân đoạn. Mặc định 60 (≈ 3 tháng giao dịch).
    crit_value : float | None
        Ngưỡng supF. None → dùng bảng Bai 1997 cho p=2, alpha=0.05.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 2 * min_seg + 1:
        return BaiPerronResult(n_breaks=0, n_obs=n)

    Xfull = _design(x)

    # Ngưỡng Bai 1997 (Table 1) cho mô hình với p=2 tham số, alpha=0.05
    if crit_value is None:
        crit_table = {1: 11.47, 2: 13.42, 3: 14.86}
        crit_value = crit_table.get(max_breaks, 14.86)

    def segment_rss(i0: int, i1: int) -> float:
        return _rss(y[i0:i1], Xfull[i0:i1])

    # DP: dp[k][j] = min RSS để chia y[0:j] thành k+1 đoạn
    candidates = list(range(min_seg, n - min_seg + 1))
    best_k = 0
    best_breaks: list[int] = []
    best_supF = 0.0

    rss_total = segment_rss(0, n)
    if rss_total <= 0:
        return BaiPerronResult(n_breaks=0, n_obs=n)

    for k in range(1, max_breaks + 1):
        # Tìm k điểm gãy tối ưu bằng dynamic programming đơn giản
        # Trạng thái: dp[j] = min RSS chia [0..j) thành m đoạn
        # Khởi tạo m=1
        dp_prev = {j: segment_rss(0, j) for j in range(min_seg, n - min_seg + 1)}
        breaks_prev: dict[int, list[int]] = {j: [] for j in dp_prev}
        for seg in range(1, k + 1):
            dp_curr: dict[int, float] = {}
            breaks_curr: dict[int, list[int]] = {}
            for j in dp_prev:
                best_rss = np.inf
                best_b: list[int] = []
                best_m = -1
                for m in range(min_seg, j - min_seg + 1):
                    if m not in dp_prev:
                        continue
                    r = dp_prev[m] + segment_rss(m, j)
                    if r < best_rss:
                        best_rss = r
                        best_b = breaks_prev[m] + [m]
                        best_m = m
                if best_m >= 0:
                    dp_curr[j] = best_rss
                    breaks_curr[j] = best_b
            dp_prev = dp_curr
            breaks_prev = breaks_curr

        # Lấy cấu hình tốt nhất cho k điểm gãy trên toàn chuỗi
        final_j = n
        if final_j not in dp_prev:
            continue
        rss_k = dp_prev[final_j]
        breaks = breaks_prev[final_j]

        # supF statistic cho k điểm gãy (ước lượng đơn giản hoá)
        # F = ((RSS_total - RSS_k) / k) / (RSS_k / (n - 2*(k+1)))
        df_num = k
        df_den = n - 2 * (k + 1)
        if df_den <= 0 or rss_k <= 0:
            continue
        supF = ((rss_total - rss_k) / df_num) / (rss_k / df_den)
        if supF > best_supF:
            best_supF = supF
            if supF > crit_value:
                best_k = k
                best_breaks = sorted(breaks)

    p_value = float(stats.f.sf(best_supF, best_k if best_k else 1, n - 2 * (best_k + 1)))
    return BaiPerronResult(
        n_breaks=best_k,
        break_indices=best_breaks,
        sup_f=float(best_supF),
        p_value=p_value,
        significant=best_k > 0,
        n_obs=n,
    )


@dataclass
class CusumResult:
    statistic: float
    p_value: float
    significant: bool
    max_pos: int
    method: str = "cusum_squares"


def cusum_squares(y: Sequence[float], x: Sequence[float], alpha: float = 0.05) -> CusumResult:
    """CUSUM of squares (Brown–Durbin–Evans 1975).

    Trả về statistic = max |S_t - t/T| scaled, vị trí max, và p-value gần đúng
    dựa trên phân phối Kolmogorov–Smirnov.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 30:
        return CusumResult(statistic=float("nan"), p_value=1.0, significant=False, max_pos=-1)

    X = _design(x)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss_total = float(resid @ resid)
    if rss_total <= 0:
        return CusumResult(statistic=float("nan"), p_value=1.0, significant=False, max_pos=-1)
    cum = np.cumsum(resid ** 2) / rss_total  # S_t theo công thức BDE
    expected = np.arange(1, n + 1) / n
    diff = cum - expected
    scaled = np.sqrt(n) * np.abs(diff)
    max_pos = int(np.argmax(scaled))
    statistic = float(scaled[max_pos])
    # xấp xỉ bằng KS: p ≈ 2 * sum (-1)^{k-1} e^{-2 k^2 (stat/n)^2 ...}
    # dùng scipy stats.kstwobign cho ĐO near-asymptotic:
    p_value = float(2 * np.exp(-2 * (statistic / np.sqrt(n)) ** 2 * n))
    p_value = min(max(p_value, 0.0), 1.0)
    # Ngưỡng Harvey-Collier xấp xỉ cho alpha=0.05
    return CusumResult(
        statistic=statistic, p_value=p_value,
        significant=statistic > 1.36 and p_value < alpha,
        max_pos=max_pos,
    )


def rolling_corr_stability(
    df: pd.DataFrame,
    s_col: str,
    t_col: str,
    group_cols: Sequence[str],
    window: int = 120,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Rolling Pearson correlation theo từng nhóm, để vẽ biểu đồ ổn định."""
    df = df.sort_values(by=list(group_cols) + ["date"] if "date" in df.columns
                        else list(group_cols))
    parts = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rc = sub[t_col].rolling(window, min_periods=min_periods).corr(sub[s_col])
        out = sub[["date"]].copy() if "date" in sub.columns else sub.index.to_frame()
        for k, v in zip(group_cols, keys):
            out[k] = v
        out["rolling_corr"] = rc.to_numpy()
        parts.append(out)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
