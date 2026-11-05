"""
Ánh xạ bài báo → mã chứng khoán bằng từ điển tên công ty (không cần NER model).

Cách hoạt động:
- Chuẩn hóa text (bỏ dấu + lowercase) rồi tìm alias trong config/ticker_map.yaml
  với kiểm tra biên từ (không khớp giữa từ) — tránh nhầm "vib" trong từ khác.
- Đồng thời quét mã CK viết hoa xuất hiện như một token riêng ("VCB").

Hạn chế minh bạch: từ điển không bắt được tên viết tắt lạ/tin không nêu tên.
Bổ sung alias vào yaml theo thực tế crawl là cách nuôi bộ matching.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .textnorm import normalize_text


def load_ticker_map(yaml_path: str | Path = "config/ticker_map.yaml") -> list[dict]:
    """Đọc yaml → [(ticker, [alias đã chuẩn hóa])]. Alias gốc giữ lại để debug."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ánh xạ: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = cfg.get("tickers", [])
    if not entries:
        raise ValueError(f"{path} thiếu key 'tickers'.")
    mapping = []
    for e in entries:
        ticker = str(e["ticker"]).upper().strip()
        aliases = {normalize_text(n) for n in e.get("names", []) if n and normalize_text(n)}
        aliases.add(ticker.lower())  # mã CK cũng là một "tên"
        mapping.append({"ticker": ticker, "aliases": aliases})
    return mapping


def _match_with_boundary(haystack: str, needle: str) -> bool:
    """Tìm `needle` trong `haystack` đã chuẩn hóa, kiểm tra biên từ hai bên."""
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return False
        left_ok = idx == 0 or not haystack[idx - 1].isalnum()
        right_ok = (idx + len(needle) == len(haystack)
                    or not haystack[idx + len(needle)].isalnum())
        if left_ok and right_ok:
            return True
        start = idx + 1


def match_tickers(text: str, mapping: list[dict]) -> list[str]:
    """Trả về danh sách mã CK (đã sort) được nhắc trong text. ''/None → []."""
    if not text or not str(text).strip():
        return []
    norm = normalize_text(text)
    found = set()
    for entry in mapping:
        for alias in entry["aliases"]:
            if alias and _match_with_boundary(norm, alias):
                found.add(entry["ticker"])
                break  # một alias khớp là đủ cho ticker này
    return sorted(found)
