"""
Dedup bài báo chéo nguồn — 2 tầng, không cần model:

1. Vân tay chính xác: md5(normalize(title) bỏ khoảng trắng) — bắt tin đăng lại
   hệt nhau trên các báo khác nhau.
2. Mờ (fuzzy): so tiêu đề mới với các tiêu đề trong cửa sổ gần nhất bằng
   difflib.SequenceMatcher (stdlib) — bắt tin "đổi miec" khi nguồn chép nhau
   ~85% nội dung tiêu đề.

Dùng streaming: `is_duplicate(new_title, recent)` trong pipeline hằng ngày,
hoặc batch: `find_duplicates(df)` cho toàn bộ lịch sử.
"""

from __future__ import annotations

import difflib
import hashlib

import pandas as pd

from .textnorm import normalize_text, title_fingerprint

DEFAULT_THRESHOLD = 0.85


def is_duplicate(new_title: str, recent_titles: list[str],
                 threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Kiểm tra 1 tiêu đề mới so với danh sách tiêu đề đã thấy (chuỗi gốc)."""
    fp = title_fingerprint(new_title)
    if not fp:
        return False
    norm_new = normalize_text(new_title)
    for t in recent_titles:
        if title_fingerprint(t) == fp:      # tầng 1: khớp chính xác
            return True
        norm_old = normalize_text(t)
        if norm_old and difflib.SequenceMatcher(None, norm_new, norm_old).ratio() >= threshold:
            return True                      # tầng 2: khớp mờ
    return False


def find_duplicates(df: pd.DataFrame, title_col: str = "title",
                    date_col: str | None = None, window_days: int | None = 14,
                    threshold: float = DEFAULT_THRESHOLD) -> pd.Series:
    """Đánh dấu trùng lặp trên DataFrame (sắp theo thời gian trước khi gọi!).

    Trả về boolean Series: True = bài bị coi là TRÙNG (bài đầu tiên của nhóm giữ lại).
    - window_days: chỉ so với các bài trong N ngày trước (None = so với tất cả trước đó).
    """
    fps: list[str] = []
    seen_exact: dict[str, int] = {}
    kept_norm: list[tuple[pd.Timestamp | None, str]] = []
    result = []

    titles = df[title_col].astype(str).tolist()
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col]).tolist()
    else:
        dates = [None] * len(df)

    for i, title in enumerate(titles):
        fp = title_fingerprint(title)
        norm = normalize_text(title)
        dup = False

        if fp in seen_exact:
            dup = True
        else:
            cur_date = dates[i]
            for j, (d_old, norm_old) in enumerate(reversed(kept_norm)):
                if window_days is not None and cur_date is not None and d_old is not None:
                    if (cur_date - d_old).days > window_days:
                        break  # kept_norm giữ thứ tự cũ→mới, hết cửa sổ thì dừng
                if difflib.SequenceMatcher(None, norm, norm_old).ratio() >= threshold:
                    dup = True
                    break

        result.append(dup)
        if not dup:
            seen_exact[fp] = i
            kept_norm.append((dates[i], norm))

    return pd.Series(result, index=df.index)
