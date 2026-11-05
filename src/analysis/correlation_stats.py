"""
Công cụ thống kê nghiêm ngặt cho phân tích tương quan S_t ↔ thị trường.

Bổ sung 3 thứ mà correlation in-sample đơn thuần thiếu:

1. Benjamini–Hochberg (FDR): đang test nhiều ticker × nhiều feature →
   5% trong số đó sẽ "đạt ý nghĩa" DO NGẪU NHIÊN. BH kiểm soát false discovery.
2. Out-of-sample stability: tương quan phải ổn định ở cả giai đoạn sau
   (30% cuối) mới đáng tin — nếu chỉ có ở 70% đầu là spurious/regime-specific.
3. Lag analysis: tin ĐĂNG KỸ trước giá mới có ý nghĩa dự báo. Nếu correlation
   mạnh nhất ở lag 0 hoặc tin sau giờ đóng cửa không được gán lại, kết quả
   nhiều khả năng là phản ánh đồng thời (concurrent), không phải dự báo.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _scipy_stats

    SCIPY_OK = True
except ImportError:  # scipy đi kèm scikit-learn — thường có sẵn
    SCIPY_OK = False


def benjamini_hochberg(pvals) -> np.ndarray:
    """Trả về q-values đã hiệu chỉnh BH (giữ thứ tự đầu vào).

    Ví dụ: 100 test p<0.05 ngẫu nhiên → BH loại gần hết, giữ đúng những cái mạnh.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # ép đơn điệu tăng từ cuối lên
    q = np.empty(n)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def _pearson_p(x: np.ndarray, y: np.ndarray):
    """Pearson + p-value; scipy có sẵn thì dùng, không thì p=None."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), None
    if SCIPY_OK:
        r, p = _scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    x, y = x - x.mean(), y - y.mean()
    r = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    return r, None


def out_of_sample_corr(x, y, train_frac: float = 0.7) -> dict:
    """Tương quan giai đoạn đầu vs giai đoạn cuối → đo độ ổn định theo thời gian."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    cut = int(len(x) * train_frac)
    r1, p1 = _pearson_p(x[:cut], y[:cut])
    r2, p2 = _pearson_p(x[cut:], y[cut:])
    stable = (not np.isnan(r1) and not np.isnan(r2)
              and np.sign(r1) == np.sign(r2) and abs(r1) > 1e-9 and abs(r2) > 1e-9
              and min(abs(r1), abs(r2)) / max(abs(r1), abs(r2)) > 0.5)
    return {"corr_train": round(r1, 4) if not np.isnan(r1) else None,
            "p_train": round(p1, 5) if p1 is not None and not np.isnan(p1) else None,
            "corr_test": round(r2, 4) if not np.isnan(r2) else None,
            "p_test": round(p2, 5) if p2 is not None and not np.isnan(p2) else None,
            "n_train": int(cut), "n_test": int(len(x) - cut),
            "stable": bool(stable)}


def lag_correlations(sentiment: np.ndarray, returns: np.ndarray,
                     lags: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Correlation giữa S_t (đã gán phiên) và return ngày t, t+lag…

    Lags > 0: sentiment ĐI TRƯỚC return lag ngày → đúng nghĩa dự báo.
    Lags < 0: return đi trước sentiment → chỉ phản ánh sự kiện (concurrent).
    """
    s = np.asarray(sentiment, float)
    r = np.asarray(returns, float)
    out = {}
    for lag in lags:
        if lag >= 0:
            x, y = s[:len(s) - lag if lag else None], r[lag:]
        else:
            x, y = s[-lag:], r[:len(r) + lag]
        mask = ~(np.isnan(x) | np.isnan(y))
        corr, p = _pearson_p(x[mask], y[mask])
        out[lag] = {"corr": round(corr, 4) if not np.isnan(corr) else None,
                    "p_value": round(p, 5) if p is not None else None,
                    "n": int(mask.sum())}
    return out
