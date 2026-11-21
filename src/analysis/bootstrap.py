"""Block bootstrap cho khoảng tin cậy của tương quan.

Lý do dùng *block* bootstrap: chuỗi tài chính có autocorrelation và volatility
clustering, nên bootstrap i.i.d. sẽ làm mất cấu trúc thời gian → CI thiên lệch.
Block bootstrap với khối dài L giữ được tương quan cục bộ.

API
---
- ``block_bootstrap_corr(x, y, L=5, n_boot=10000, alpha=0.05, method="pearson")``
    Trả về dict {point, lo, hi, n_boot, method, block_size}.
- ``bootstrap_corr_table(df, value_cols, pair_cols, ...)``
    Trả về DataFrame mỗi dòng = 1 cặp (ticker, feature) + cột ci_lo, ci_hi.

Lựa chọn block size mặc định L=5 (5 ngày giao dịch ~ 1 tuần) theo khuyến nghị
Politis & White (2004) cho dữ liệu tài chính daily. Có thể tự động hoá bằng
``optimal_block_size`` từ package `arch.bootstrap` nếu đã cài.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class BootResult:
    point: float
    lo: float
    hi: float
    n_boot: int
    method: str
    block_size: int
    n_used: int

    def as_dict(self) -> dict:
        return self.__dict__


def _moving_block_indices(n: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Lấy các chỉ số khối di động (circular block bootstrap nếu thiếu)."""
    if L <= 0 or L > n:
        L = max(1, min(L, n))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + L) for s in starts])
    return idx[:n]


def block_bootstrap_corr(
    x: Sequence[float],
    y: Sequence[float],
    L: int = 5,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    method: str = "pearson",
    seed: int | None = 42,
) -> BootResult:
    """Block bootstrap CI cho hệ số tương quan giữa x và y.

    Parameters
    ----------
    x, y : sequence
        Hai chuỗi số cùng độ dài (theo thứ tự thời gian).
    L : int
        Kích thước khối (block size), đơn vị = quan sát. Mặc định 5.
    n_boot : int
        Số lần resample. Mặc định 10.000 — đủ cho CI 95% ổn định ở chữ số thứ 2.
    alpha : float
        Mức ý nghĩa, mặc định 0.05 → CI 95%.
    method : str
        ``pearson`` hoặc ``spearman``.
    seed : int | None
        Seed để tái lập. None = ngẫu nhiên.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2 * L:
        raise ValueError(f"Quá ít quan sát hợp lệ (n={n}) cho block size L={L}.")

    rng = np.random.default_rng(seed)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if method == "pearson":
            if np.std(a) == 0 or np.std(b) == 0:
                return np.nan
            return float(np.corrcoef(a, b)[0, 1])
        elif method == "spearman":
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            if np.std(ra) == 0 or np.std(rb) == 0:
                return np.nan
            return float(np.corrcoef(ra, rb)[0, 1])
        else:
            raise ValueError(f"method phải là 'pearson' hoặc 'spearman', nhận '{method}'")

    point = _corr(x, y)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _moving_block_indices(n, L, rng)
        boots[b] = _corr(x[idx], y[idx])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        raise ValueError("Toàn bộ resample trả NaN — kiểm tra variance của x, y.")
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return BootResult(
        point=point, lo=lo, hi=hi, n_boot=len(boots),
        method=method, block_size=L, n_used=n,
    )


def bootstrap_corr_table(
    df: pd.DataFrame,
    pair_cols: Sequence[str],
    value_cols: Sequence[str],
    L: int = 5,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    method: str = "pearson",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Chạy block bootstrap cho nhiều cặp (theo pair_cols) và 1 cặp giá trị.

    Parameters
    ----------
    df : DataFrame
        Phải chứa các cột trong ``pair_cols`` (định danh, vd ['ticker']) và
        ``value_cols`` (2 cột số, vd ['S_t', 'avg_return']).
    pair_cols : list[str]
        Cột định danh nhóm (vd ['ticker'] hoặc ['ticker','feature']).
    value_cols : list[str]
        Đúng 2 cột: [sentiment_col, target_col].
    """
    if len(value_cols) != 2:
        raise ValueError("value_cols phải có đúng 2 cột.")
    s_col, t_col = value_cols
    rows = []
    for keys, sub in df.groupby(list(pair_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        res = block_bootstrap_corr(
            sub[s_col].to_numpy(), sub[t_col].to_numpy(),
            L=L, n_boot=n_boot, alpha=alpha, method=method, seed=seed,
        )
        row = dict(zip(pair_cols, keys))
        row.update({
            "s_col": s_col, "t_col": t_col,
            "corr": res.point, "ci_lo": res.lo, "ci_hi": res.hi,
            "n_boot": res.n_boot, "n_obs": res.n_used,
            "block_size": res.block_size, "method": res.method,
            "significant": bool(res.lo > 0 or res.hi < 0),
        })
        rows.append(row)
    return pd.DataFrame(rows)
