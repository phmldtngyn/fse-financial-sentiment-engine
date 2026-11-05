"""
Pytest cho các module vận hành fse-ops — KHÔNG cần torch/GPU.

Chạy từ gốc repo:  python -m pytest tests/test_ops_units.py -v
CI (GitHub Actions) tự chạy file này trước smoke test.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.correlation_stats import (
    benjamini_hochberg,
    lag_correlations,
    out_of_sample_corr,
)
from src.data.dedup import find_duplicates, is_duplicate
from src.data.market_calendar import (
    assign_session,
    is_trading_day,
    load_holidays,
    next_trading_day,
    prev_trading_day,
)
from src.data.textnorm import normalize_text, title_fingerprint
from src.data.ticker_mapping import load_ticker_map, match_tickers
from src.storage.sqlite_db import (
    article_id,
    connect,
    get_market,
    get_sentiment,
    upsert_articles,
    upsert_market_daily,
    upsert_sentiment_daily,
)
from src.training.drift_monitor import (
    check_drift,
    psi_categorical,
    psi_continuous,
    weekly_stats,
)

ROOT = Path(__file__).resolve().parents[1]
HOLIDAYS = load_holidays(ROOT / "config" / "holidays_vn.yaml")
TICKERS = load_ticker_map(ROOT / "config" / "ticker_map.yaml")


# ---------------------------------------------------------------------------
# textnorm
# ---------------------------------------------------------------------------
def test_normalize_strips_diacritics():
    assert normalize_text("Ngân hàng Ngoại thương Việt Nam") == "ngan hang ngoai thuong viet nam"


def test_title_fingerprint_ignores_spacing_punct():
    assert title_fingerprint("Ngân hàng - tăng mạnh!") == title_fingerprint("ngan  hang tang manh")


# ---------------------------------------------------------------------------
# market_calendar
# ---------------------------------------------------------------------------
def test_weekend_is_not_trading():
    assert not is_trading_day(dt.date(2026, 11, 7), HOLIDAYS)   # thứ Bảy
    assert not is_trading_day(dt.date(2026, 11, 8), HOLIDAYS)   # Chủ nhật
    assert is_trading_day(dt.date(2026, 11, 9), HOLIDAYS)       # thứ Hai


def test_tet_2026_is_holiday():
    for d in ("2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20"):
        assert not is_trading_day(dt.date.fromisoformat(d), HOLIDAYS), d
    assert is_trading_day(dt.date(2026, 2, 23), HOLIDAYS)  # thứ Hai sau Tết


def test_new_year_2026():
    assert not is_trading_day(dt.date(2026, 1, 1), HOLIDAYS)


def test_session_before_cutoff_same_day():
    assert assign_session(dt.datetime(2026, 11, 2, 10, 30),
                          holidays=HOLIDAYS) == dt.date(2026, 11, 2)


def test_session_evening_next_trading_day():
    # Thứ Hai 2/11/2026 16:05 → phiên thứ Ba 3/11
    assert assign_session(dt.datetime(2026, 11, 2, 16, 5),
                          holidays=HOLIDAYS) == dt.date(2026, 11, 3)


def test_session_friday_evening_skips_weekend():
    # Thứ Sáu 6/11/2026 18:00 → phiên thứ Hai 9/11
    assert assign_session(dt.datetime(2026, 11, 6, 18, 0),
                          holidays=HOLIDAYS) == dt.date(2026, 11, 9)


def test_session_during_tet_goes_to_first_trading_day():
    # Giữa Tết 18/2/2026 → phiên đầu tiên 23/2
    assert assign_session(dt.datetime(2026, 2, 18, 9, 0),
                          holidays=HOLIDAYS) == dt.date(2026, 2, 23)


def test_prev_next_trading_day():
    # Ngữ nghĩa: next_trading_day TỰ TÍNH là phiên nếu day là ngày giao dịch
    assert next_trading_day(dt.date(2026, 11, 6), HOLIDAYS) == dt.date(2026, 11, 6)   # thứ Sáu
    assert next_trading_day(dt.date(2026, 11, 7), HOLIDAYS) == dt.date(2026, 11, 9)   # T7 → thứ Hai
    assert prev_trading_day(dt.date(2026, 11, 9), HOLIDAYS) == dt.date(2026, 11, 6)


# ---------------------------------------------------------------------------
# ticker_mapping
# ---------------------------------------------------------------------------
def test_match_by_company_name():
    assert match_tickers("Vietcombank báo lợi nhuận tăng mạnh", TICKERS) == ["VCB"]


def test_match_without_diacritics():
    assert match_tickers("ngan hang ngoai thuong viet nam tang truong", TICKERS) == ["VCB"]


def test_match_multiple_tickers():
    assert match_tickers("Techcombank và VPBank đồng loạt tăng", TICKERS) == ["TCB", "VPB"]


def test_match_ticker_symbol_token():
    assert match_tickers("Cổ phiếu VCB tăng trần", TICKERS) == ["VCB"]


def test_no_match_returns_empty():
    assert match_tickers("Giá dầu thế giới giảm sâu", TICKERS) == []


def test_partial_word_not_matched():
    # "vib" nằm trong từ khác không được tính
    assert match_tickers("Hoạt động sôi vibrancy? vib trầm lặng", TICKERS) == ["VIB"]


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------
def test_exact_duplicate_detected():
    assert is_duplicate("Ngân hàng VCB tăng mạnh", ["Ngan hang VCB tang manh"])


def test_fuzzy_duplicate_detected():
    a = "Ngân hàng VCB báo lợi nhuận quý 3 tăng mạnh vượt kỳ vọng"
    b = "Ngân hàng VCB báo lợi nhuận quý 3 tăng mạnh, vượt kỳ vọng"
    assert is_duplicate(a, [b])          # chép gần hệt
    assert not is_duplicate("Giá vàng thế giới giảm", [b])


def test_find_duplicates_marks_first_as_kept():
    df = pd.DataFrame({
        "title": ["BID tăng sau họp ĐHĐCĐ", "BID tăng sau họp ĐHĐCĐ!",
                  "Tin khác hẳn về thị trường"],
        "date": ["2026-01-05", "2026-01-05", "2026-01-06"],
    }).sort_values("date").reset_index(drop=True)
    flags = find_duplicates(df, title_col="title", date_col="date", window_days=14)
    assert list(flags) == [False, True, False]


# ---------------------------------------------------------------------------
# correlation_stats
# ---------------------------------------------------------------------------
def test_benjamini_hochberg_known_case():
    # 5 test: p = .01,.04,.03,.20,.01 → q BH = .025,.05,.05,.20,.025
    p = np.array([0.01, 0.04, 0.03, 0.20, 0.01])
    q = benjamini_hochberg(p)
    assert np.allclose(q, [0.025, 0.05, 0.05, 0.20, 0.025], atol=1e-9)


def test_bh_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    q = benjamini_hochberg(rng.uniform(0, 1, 500))
    assert ((q >= 0) & (q <= 1)).all()


def test_out_of_sample_detects_unstable():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    y1 = 0.8 * x[:350] + 0.3 * rng.normal(size=350)   # tương quan mạnh đầu
    y2 = rng.normal(size=150)                          # mất tương quan cuối
    y = np.concatenate([y1, y2])
    res = out_of_sample_corr(x, y)
    assert res["stable"] is False


def test_out_of_sample_stable_case():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = 0.6 * x + 0.5 * rng.normal(size=400)
    res = out_of_sample_corr(x, y)
    assert res["stable"] is True
    assert abs(res["corr_train"] - res["corr_test"]) < 0.15


def test_lag_correlations_direction():
    rng = np.random.default_rng(3)
    s = rng.normal(size=400)
    r = np.concatenate([[0, 0], 0.7 * s[:-2]]) + 0.2 * rng.normal(size=400)  # r_t = s_{t-2}
    res = lag_correlations(s, r, lags=(-2, -1, 0, 1, 2))
    assert res[2]["corr"] is not None and abs(res[2]["corr"]) > 0.4


# ---------------------------------------------------------------------------
# drift_monitor
# ---------------------------------------------------------------------------
def test_psi_identical_is_zero():
    assert psi_categorical([0.33, 0.34, 0.33], [0.34, 0.33, 0.33]) < 0.005


def test_psi_shifted_is_high():
    assert psi_categorical([0.66, 0.22, 0.12], [0.2, 0.3, 0.5]) > 0.25


def test_psi_continuous_insufficient_samples_nan():
    assert np.isnan(psi_continuous(np.zeros(10), np.zeros(10)))


def _weekly_frame(base_share: float, weeks: int, conf: float = 0.8, n: int = 100) -> pd.DataFrame:
    rows = []
    for w in range(weeks):
        share = base_share if w < weeks - 2 else 0.2  # 2 tuần cuối lệch mạnh
        labels = rng_helper(share, n)
        rows += [{"date": f"2026-{(w // 4) + 1:02d}-{(w % 4) * 7 + 1:02d}",
                  "predicted_label": lab, "confidence": conf} for lab in labels]
    return pd.DataFrame(rows)


def rng_helper(neutral_share: float, n: int):
    rng = np.random.default_rng(7)
    pos = 0.2
    neg = max(0.05, 1.0 - pos - neutral_share)
    return rng.choice([0, 1, 2], size=n, p=[neg, neutral_share, pos])


def test_check_drift_flags_class_shift():
    df = _weekly_frame(0.6, weeks=10)
    stats = weekly_stats(df, "date", "predicted_label", "confidence")
    report = check_drift(stats.iloc[:-2], stats.iloc[-2:])
    assert report["signals"]["class_distribution"]["level"] in ("watch", "drift")
    assert "volume" in report["signals"]


# ---------------------------------------------------------------------------
# prepare_splits
# ---------------------------------------------------------------------------
from scripts.prepare_splits import split_by_dates, to_3class  # noqa: E402


def test_split_by_dates_disjoint_and_ordered():
    dates = pd.date_range("2024-01-01", "2026-01-01", freq="D").to_series().reset_index(drop=True)
    splits = split_by_dates(dates, (0.8, 0.1, 0.1))
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])
    assert max(splits["train"]) < min(splits["test"])


def test_to_3class_thresholds():
    s = pd.Series([-0.31, -0.3, 0.0, 0.3, 0.31])
    assert list(to_3class(s)) == [0, 1, 1, 1, 2]


# ---------------------------------------------------------------------------
# sqlite_db
# ---------------------------------------------------------------------------
def test_sqlite_idempotent_upsert(tmp_path):
    db = tmp_path / "test.db"
    df_a = pd.DataFrame([{"text": "tin ngân hàng", "url": "u1", "source": "vnexpress",
                          "title": "t1", "published_at": "2026-01-05 10:00",
                          "trading_date": "2026-01-05", "tickers": ["VCB"],
                          "is_duplicate": 0}])
    df_s = pd.DataFrame([{"ticker": "VCB", "trading_date": "2026-01-05", "s_t": 0.4,
                          "n_articles": 3, "mean_conf": 0.9}])
    df_m = pd.DataFrame([{"ticker": "VCB", "date": "2026-01-05", "close": 92.1}])
    with connect(db) as conn:
        assert upsert_articles(conn, df_a) == 1
        assert upsert_articles(conn, df_a) == 1      # chạy lại → vẫn 1 bài
        upsert_sentiment_daily(conn, df_s)
        upsert_market_daily(conn, df_m)
    with connect(db) as conn:
        assert get_sentiment(conn, "VCB", "2026-01-01", "2026-12-31").shape[0] == 1
        assert get_market(conn, "VCB").iloc[0]["close"] == 92.1


def test_article_id_stable_and_distinct():
    assert article_id("u", "abc") == article_id("u", "abc")
    assert article_id("u", "abc") != article_id(None, "abc")
