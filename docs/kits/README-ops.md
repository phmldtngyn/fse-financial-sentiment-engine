# FSE Ops Kit — Bộ vận hành tự động (đợt 2)

Bổ sung cho `fse-training` (đợt 1): dữ liệu & crawler, drift + active learning,
hạ tầng + CI, correlation nghiêm ngặt, SQLite storage, license. **Đã kiểm chứng:
33/33 test pass** (unit + chạy thật từng luồng CLI).

> Quyết định RAG trong đợt này: **NGỪNG sử dụng** — RAG không nằm trên đường
> sentiment → S_t → correlation, index hiện tại build từ ViFiD + ngày đăng GÁN GIẢ.
> Giữ code, bỏ mount `data/rag_index` và route `/rag` khỏi compose. Rebuild sau
> (nếu cần) trên tin thật đã crawl + ngày đăng thật.

---

## 1. Bản đồ file: `fse-ops/` → repo gốc

| File | Đặt vào | Ghi đè? |
|---|---|---|
| `src/data/textnorm.py`, `market_calendar.py`, `ticker_mapping.py`, `dedup.py` | `src/data/` | 🆕 |
| `src/training/drift_monitor.py` | `src/training/` (cạnh file đợt 1) | 🆕 |
| `src/analysis/correlation_stats.py` | `src/analysis/` (thư mục mới) | 🆕 |
| `src/storage/sqlite_db.py` | `src/storage/` (thư mục có sẵn) | 🆕 |
| `scripts/prepare_splits.py`, `apply_news_alignment.py`, `export_for_annotation.py`, `grade_annotation_agreement.py`, `correlation_robust.py`, `init_db.py`, `migrate_parquet_to_sqlite.py`, `scheduled_jobs.py` | `scripts/` | 🆕 |
| `config/holidays_vn.yaml`, `ticker_map.yaml`, `scheduler.yaml` | `config/` | 🆕 |
| `docker-compose.yml` | gốc repo | ✅ thay bản cũ (sửa healthcheck + thêm scheduler) |
| `.env.example`, `LICENSE`, `requirements-ops.txt` | gốc repo | 🆕 |
| `.github/workflows/ci.yml` | `.github/workflows/` (tạo mới) | 🆕 |
| `tests/test_ops_units.py`, `test_cli_flows.py` | `tests/` (đợt 1 đã có) | 🆕 |

```bash
cd fse-financial-sentiment-engine && SRC=../fse-ops
cp -r $SRC/src/data/*.py src/data/
mkdir -p src/analysis && cp $SRC/src/analysis/*.py src/analysis/
cp $SRC/src/training/drift_monitor.py src/training/
cp $SRC/src/storage/sqlite_db.py src/storage/
cp $SRC/scripts/*.py scripts/
cp $SRC/config/*.yaml config/
cp $SRC/docker-compose.yml $SRC/.env.example $SRC/LICENSE $SRC/requirements-ops.txt .
mkdir -p .github/workflows && cp $SRC/.github/workflows/ci.yml .github/workflows/
cp $SRC/tests/*.py tests/
pip install -r requirements-ops.txt
```

## 2. Thứ tự triển khai khuyến nghị

### Bước 1 — Chặn leakage: split theo thời gian (chạy 1 lần)
```bash
# Cần parquet nguồn CÓ cột date (nếu chưa có → build trước, script sẽ dừng và nhắc)
python -m scripts.prepare_splits \
    --source data/raw/vn_quant_fulltext.parquet \
    --score-col Sentiment_Score --date-col date
# → 6 file: {train,val,test}.parquet + {train,val,test}_3class.parquet (giữ cột date)
```
Sau đó đặt `data.date_col: date` trong `config/train_config.yaml` → lần train sau
`verify_splits` tự kiểm tra leakage thời gian.

### Bước 2 — Chuẩn hóa tin thô (chạy mỗi ngày)
```bash
python -m scripts.apply_news_alignment \
    --input data/parsed/articles.parquet --to-sqlite data/fse.db
```
Tự động: ticker mapping (16 mã ngân hàng, chống dấu/không dấu) + gán phiên giao
dịch (tin sau 15:00/T7/CN/lễ → phiên kế) + dedup chéo nguồn (vân tay + fuzzy 85%).
Bổ sung alias vào `config/ticker_map.yaml` khi thấy nhiều bài "không khớp ticker".

### Bước 3 — SQLite (thay file parquet cho luồng chạy liên tục)
```bash
python -m scripts.init_db --db data/fse.db --migrate   # tạo schema + migrate parquet cũ
```
`src/storage/sqlite_db.py`: WAL mode, upsert idempotent (chạy lại không nhân đôi),
bảng `articles / sentiment_daily / market_daily / pipeline_runs`. Khi cần Postgres,
chỉ thay file này — interface upsert/get giữ nguyên.

### Bước 4 — Drift monitor + active learning (theo tuần/tháng)
```bash
python -m src.training.drift_monitor \
    --input data/processed/predictions_daily.parquet --fail-on-drift   # cron tuần
python -m scripts.export_for_annotation \
    --checkpoint checkpoints/production/best.pt \
    --news data/processed/articles_aligned.parquet --n 500             # mỗi tháng
# (người gán nhãn điền human_label trong xlsx)
python -m scripts.grade_annotation_agreement --file data/gold/review_batch_*.xlsx
# → Cohen's kappa: >0.6 khá tốt, >0.8 rất tốt; <0.4 → rà lại tiêu chí gán nhãn
```

### Bước 5 — Correlation nghiêm ngặt (chạy lại analysis cũ qua bộ lọc)
```bash
python -m scripts.correlation_robust \
    --input data/processed/correlation/correlation_results.parquet
# → cột q_value_bh + báo cáo: bao nhiêu quan hệ "đạt ý nghĩa" chỉ do trùng hợp ngẫu nhiên
```
Thêm: `out_of_sample_corr(x, y)` (tương quan có ổn định nửa sau không?) và
`lag_correlations(s_t, returns, lags=(0,1,2))` (tin ĐI TRƯỚC giá mới là dự báo).

### Bước 6 — Lên container
```bash
cp .env.example .env
# Thêm route /health vào api/main.py (2 dòng):
#   @app.get("/health")\nasync def health(): return {"status": "ok"}
docker compose up -d   # api (có healthcheck) + dashboard + scheduler
```
Scheduler chạy theo `config/scheduler.yaml`: alignment 19:00 hằng ngày, drift
sáng T7, retrain (orchestrator champion/challenger) ngày 1 hằng tháng. Tắt/bật
job bằng `enabled: true/false` — không sửa code. `daily_pipeline` đang `false`
vì `run_daily_pipeline.py` gốc vẫn là stub.

## 3. Giả định đã ghi rõ (chỉnh được trong code/config)

| Giả định | Giá trị | Chỉnh ở |
|---|---|---|
| Giờ cắt tin sang phiên kế | 15:00 (ATC đóng cửa) | `assign_session(cutoff_hour=)` |
| Thứ Bảy | Không giao dịch, kể cả khi Nhà nước hoán đổi | đúng thực tế HOSE |
| Ngày nghỉ lễ | 2016–2026 (theo công bố Bộ Nội vụ), Tết 16–20/2/2026 đã đối chiếu | `config/holidays_vn.yaml` |
| Ngưỡng dedup | Fuzzy 85% / cửa sổ 14 ngày | `find_duplicates(...)` |
| Cảnh báo drift | PSI > 0.25 = drift | `drift_monitor.PSI_WATCH` |
| Margin FDR | α = 0.05, Benjamini–Hochberg | `--alpha` |

## 4. Testing

```bash
python -m pytest tests/ -v          # 33 test: unit + luồng CLI end-to-end
python -m tests.smoke_test          # (đợt 1) pipeline training end-to-end
```
CI (`.github/workflows/ci.yml`) tự chạy cả hai khi push/PR: cài torch CPU →
pytest → smoke test → kiểm tra import module ops.

## 5. Còn lại sau đợt này (không thuộc phạm vi đã chọn)

- Gây thay thế `create_dummy_gold.py` / `create_dummy_market.py` bằng dữ liệu thật
- API auth (bạn chọn "chưa cần" — khi expose ra internet hãy bật tối thiểu API key)
- `run_daily_pipeline.py` gốc vẫn là stub: ghép crawl → align → suy luận → S_t
  → SQLite thành 1 entry point rồi bật job `daily_pipeline` trong scheduler.yaml
