"""FSE Engine — mở rộng phân tích chiều sâu.

Các module bổ sung cho bài nghiên cứu:
- ``bootstrap``     : block bootstrap CI cho tương quan
- ``breaks``        : kiểm định điểm gãy cấu trúc (Bai–Perron + CUSUM)
- ``causality``     : Granger và Toda–Yamamoto
- ``irf``           : Local Projections (Jordà 2005) cho impulse response
- ``garch_x``       : EGARCH-X với sentiment làm biến ngoại sinh
- ``partial_corr``  : tương quan riêng kiểm soát macro

Tất cả đều nhận pandas DataFrame, trả về DataFrame/Series có thể ghi parquet/csv
để pipeline downstream dùng tiếp.
"""
from . import bootstrap, breaks, causality, irf, garch_x, partial_corr  # noqa: F401

__all__ = ["bootstrap", "breaks", "causality", "irf", "garch_x", "partial_corr"]
