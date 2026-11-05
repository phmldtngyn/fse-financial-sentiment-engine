"""
Lịch giao dịch chứng khoán Việt Nam + quy tắc gán tin vào phiên giao dịch.

Quy tắc (giả định được ghi rõ, chỉnh qua tham số):
- HOSE/HNX chỉ giao dịch Thứ 2–6, KHÔNG mở thứ Bảy kể cả khi Nhà nước hoán đổi
  ngày làm việc → lịch = ngày trong tuần trừ ngày lễ trong config/holidays_vn.yaml.
- Tin đăng TRƯỚC 15:00 (giờ đóng cửa ATC ~15:00) của ngày giao dịch → thuộc phiên HÔM ĐÓ.
- Tin đăng SAU 15:00, hoặc thứ Bảy/CN/ngày lễ → thuộc phiên GIAO DỊCH KẾ TIẾP.

Lý do: giá đóng cửa đã cố định sau 15:00, tin buổi tối thường phản ánh vào
phiên mai — nếu không gán lại thì correlation S_t vs return sẽ bị lệch pha.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

_HOLIDAYS: set[str] | None = None


def load_holidays(yaml_path: str | Path = "config/holidays_vn.yaml") -> set[dt.date]:
    """Đọc danh sách ngày nghỉ lễ (gộp cả key `holidays` lẫn các key `tet_*`)."""
    global _HOLIDAYS
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file lịch nghỉ: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    days: set[dt.date] = set()
    for key, values in cfg.items():
        if not isinstance(values, list):
            continue
        for v in values:
            if isinstance(v, str):
                days.add(dt.date.fromisoformat(v.strip()))
    if not days:
        raise ValueError(f"{path} không chứa ngày nghỉ nào — kiểm tra cấu trúc yaml.")
    _HOLIDAYS = days
    return days


def is_trading_day(day: dt.date, holidays: set[dt.date] | None = None) -> bool:
    """Ngày giao dịch = trong tuần (T2–T6) và không phải ngày nghỉ lễ."""
    if holidays is None:
        holidays = _HOLIDAYS if _HOLIDAYS is not None else load_holidays()
    return day.weekday() < 5 and day not in holidays


def next_trading_day(day: dt.date, holidays: set[dt.date] | None = None) -> dt.date:
    """Phiên giao dịch kế tiếp sau `day` (bao gồm chính `day` nếu là ngày giao dịch)."""
    cur = day
    for _ in range(60):  # chặn vòng lặp khi yaml thiếu lễ của năm tương lai
        if is_trading_day(cur, holidays):
            return cur
        cur += dt.timedelta(days=1)
    raise RuntimeError(f"Không tìm thấy ngày giao dịch sau {day} trong 60 ngày — "
                       "kiểm tra lại holidays_vn.yaml (thiếu lễ của năm này?).")


def prev_trading_day(day: dt.date, holidays: set[dt.date] | None = None) -> dt.date:
    """Phiên giao dịch liền trước `day` (không tính `day`)."""
    cur = day - dt.timedelta(days=1)
    for _ in range(60):
        if is_trading_day(cur, holidays):
            return cur
        cur -= dt.timedelta(days=1)
    raise RuntimeError(f"Không tìm thấy ngày giao dịch trước {day} trong 60 ngày.")


def assign_session(published_at, cutoff_hour: int = 15, cutoff_minute: int = 0,
                   holidays: set[dt.date] | None = None) -> dt.date:
    """Gán thời điểm đăng tin → NGÀY PHIÊN GIAO DỊCH tương ứng.

    - Ngày giao dịch + đăng ≤ 15:00 → chính hôm đó.
    - Ngày giao dịch + đăng > 15:00, hoặc T7/CN/lễ → phiên kế tiếp.
    Chấp nhận datetime, date, hoặc chuỗi ISO ("2026-03-05 18:23:00").
    """
    if isinstance(published_at, str):
        published_at = dt.datetime.fromisoformat(published_at.strip())
    if isinstance(published_at, dt.datetime):
        day, time = published_at.date(), published_at.time()
    elif isinstance(published_at, dt.date):
        # chỉ có ngày, không có giờ → coi như đăng trong phiên hôm đó
        day, time = published_at, dt.time(10, 0)
    else:
        raise TypeError(f"published_at không hợp lệ: {type(published_at)}")

    cutoff = dt.time(cutoff_hour, cutoff_minute)
    if is_trading_day(day, holidays) and time <= cutoff:
        return day
    return next_trading_day(day + dt.timedelta(days=1), holidays)
