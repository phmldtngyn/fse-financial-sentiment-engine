"""Chuẩn hóa văn bản tiếng Việt dùng chung cho matching (ticker/dedup)."""

from __future__ import annotations

import re
import unicodedata


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt: 'Ngân hàng' → 'ngan hang' (NFKD + bỏ combining marks)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """Chuẩn hóa tối thiểu để so khớp: bỏ dấu + lowercase + bỏ dấu câu + gọn khoảng trắng."""
    t = strip_diacritics(str(text)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_fingerprint(title: str) -> str:
    """Vân tay tiêu đề cho dedup: chuẩn hóa + bỏ toàn bộ khoảng trắng."""
    return normalize_text(title).replace(" ", "")
