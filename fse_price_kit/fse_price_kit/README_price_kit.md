# FSE Price Kit — Lấy giá thực 26 mã ngân hàng từ Vietstock

Bộ 2 script bổ sung cho repo `fse-financial-sentiment-engine`, cho phép lấy
**giá thực (OHLCV)** từ trang công khai **Vietstock Finance** cho đúng nhóm
26 mã ngân hàng đang dùng trong nghiên cứu, và tổng hợp thành bảng daily
**khop 100% schema** với `banking_daily.parquet` cũ.

## 26 mã được lấy (trùng extract_banking_daily.py)

VCB, BID, CTG, TCB, VPB, MBB, ACB, STB, SHB, HDB, LPB, TPB, VIB, MSB, OCB,
SSB, EIB, NAB, BVB, SGB, VBB, KLB, PGB, ABB, BAB, PVCB
(hỗ trợ cả HOSE lẫn HNX/UPCOM — script tự nhận từ field `Exchange`).

## Cài đặt (máy Windows, venv đang bật)

```powershell
pip install requests pandas pyarrow
```

(pandas/pyarrow bạn đã có từ trước khi train + đọc parquet.)

## Cách chạy

Đặt 2 file vào thư mục `scripts/` của repo, rồi:

```powershell
# Bước 1 — lấy giá (26 mã, từ 2022-01-01 đến hôm nay, ~1.5 phút)
python scripts/fetch_price_vietstock.py

# Chỉ định khác:
python scripts/fetch_price_vietstock.py --start 2020-01-01
python scripts/fetch_price_vietstock.py --tickers VCB,BID --end 2026-04-30

# Bước 2 — tổng hợp thành bảng ngành
python scripts/build_banking_daily.py
```

## Đầu ra

| File | Nội dung |
|---|---|
| `data/raw/vietstock/{TICKER}.csv` | Giá từng mã: ticker, date, open, high, low, close, volume, exchange, pct |
| `data/processed/vietstock_price_daily.parquet` | Gop 26 mã (bảng giá thô dùng chung) |
| `data/processed/banking_daily_vietstock.parquet` | Bảng ngành: trade_date, total_volume, avg_return, avg_range, avg_close, n_banks |
| `*.csv` bản gop | Bản UTF-8-BOM mở Excel không lỗi font |

Quy ước khớp DB gốc (`UNIFIED_RESEARCH_MATRIX`): giá theo **nghìn đồng**,
`pct = close/prev_close − 1` (phân số), lọc ngày có ≥ 5 mã giao dịch.

## ĐÃ KIỂM CHỨNG (2026-11)

Chạy thử VCB/BID/BAB (03–04/2026) rồi đối chiếu với DB gốc:

```
VCB 2026-04-14: Vietstock (open 59.7 high 59.8 low 59.1 close 59.3
                vol 5,548,100 pct 0.001689)
             == DB UNIFIED_RESEARCH_MATRIX (đúng từng số)
VCB 2026-04-08..13: khớp 100% cả 5 phiên
```

=> Nguồn giá Vietstock có thể dùng để **kéo dài/gia hạn** dataset giá của
dự án (DB gốc chỉ phủ ~4 năm, bảng tên `hose_4years_market_data`).

## Cách hoạt động (để cit / bảo vệ đồ án)

1. `GET https://finance.vietstock.vn/{MA}/lich-su-gia.htm` → nhận cookie
   session + `__RequestVerificationToken`.
2. `POST https://finance.vietstock.vn/data/getpricehistory` với JSON
   `{Code, FromDate, ToDate, PageSize: 2000, __RequestVerificationToken}`
   → trả mảng dòng giá mới nhất trước (TradingDate là epoch ms).
3. Lọc theo năm (FromDate/ToDate), tự refresh token khi 401/403,
   retry 3 lần, giới hạn 1 request/giây.

## Tích hợp vào pipeline tương quan (K3/K4)

Sửa 1 dòng trong `scripts/join_banking_sentiment.py`:

```python
BANKING_PATH = Path("data/processed/banking_daily_vietstock.parquet")
```

rồi chạy lại join → có `banking_with_sentiment` trên dữ liệu giá MỚI, ghép
với S_t của model để cập nhật K4 mỗi khi cần. Muốn cập nhật tự động hàng
ngày sau phiên (≈18:00), thêm 2 lệnh trên vào `scripts/scheduled_jobs.py`
giống như phần tin tức.

## Ghi chú

- Script chủ động tôn trọng server: sleep 1s/request, 26 mã × vài chunk
  năm → chỉ ~30-60 request mỗi lần chạy. Đừng giảm `--sleep` xuống 0.
- Vietstock là dữ liệu công khai phục vụ nghiên cứu — không dùng cho mục
  đích thương mại/redistribute quy mô lớn.
- Nếu Vietstock đổi API (404 giữa chừng): chạy lại, báo lỗi cho tôi để
  cập nhật endpoint; phương án dự phòng là nguồn CafeF DataAPI (cùng
  chuẩn số liệu, tôi đã test endpoint sống).
