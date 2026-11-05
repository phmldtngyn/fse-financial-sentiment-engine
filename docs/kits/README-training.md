# FSE Training Pipeline — Bản chuẩn hóa & tự động hóa

Bộ code viết lại **toàn bộ giai đoạn training** của `fse-financial-sentiment-engine`
sau khi audit phát hiện các lỗi P0/P1. Đã kiểm chứng bằng smoke test end-to-end
(25/25 PASS — chạy thật trên CPU, không cần GPU, không cần tải PhoBERT).

---

## 1. Các lỗi đã sửa

| # | Lỗi của bản cũ | Đã sửa ở đâu |
|---|---|---|
| 1 | **Không set seed** → mỗi lần chạy một kết quả, không thể so sánh các lần retrain | `src/training/seed.py` + `train_fse_sft.py` (set_seed đầy đủ: random/numpy/torch/CUDA/cudnn) |
| 2 | **Class weights hardcode** `[2.75, 0.50, 1.54]` trong code | `src/training/losses.py` → `compute_class_weights()` tính động từ tập train (nghịch đảo tần suất — chính công thức sinh ra bộ số cũ) |
| 3 | **Config có 2 block mâu thuẫn**, block `fse_model` (`lambda_kd`, `alpha_loss`, `beta_loss`) là rác của pipeline distillation đã bị gỡ | `config/train_config.yaml` — một block duy nhất, đã xóa sạch |
| 4 | **Không chạm test set** khi train/eval | `src/training/evaluate.py` + `train_fse_sft.py` — eval test bắt buộc, xuất `eval_report.json` + `.md` (per-class + confusion matrix) |
| 5 | **`padding="max_length"` cứng 256** — lãng phí compute với câu 50–800 ký tự | `src/data/dataset.py` → `Collator` pad động theo độ dài lớn nhất từng batch |
| 6 | **Chạy fp32** trên GPU chậm | `train_fse_sft.py` — AMP (mixed precision), tự tắt trên CPU |
| 7 | **Early stopping không có patience** | `train_fse_sft.py` — patience 3 + min_delta 0.001 (cấu hình được) |
| 8 | **Không phát hiện text chưa tách từ** — PhoBERT yêu cầu input dạng `ngan_hang`, lỗi này giảm F1 âm thầm | `src/data/dataset.py` — `looks_segmented()` tự phát hiện → tự tách từ bằng underthesea, **có cache ra đĩa** (retrain sau chỉ xử lý câu mới) |
| 9 | **Không xác minh split theo thời gian / leakage** | `src/data/dataset.py` → `verify_splits()` — cảnh báo nếu `max(train.date) > min(test.date)`, nếu có câu trùng lặp chéo train/test, nếu lớp nào trong test quá mỏng |
| 10 | **Không tracking, print() rời rạc, checkpoint "mù"** | MLflow (parent run + child run mỗi seed), logging có cấu trúc, `best.pt` kèm `meta.json` (seed, data_hash, git_hash, config snapshot, metrics) |
| 11 | **Không có vòng lặp retrain tự động** | `scripts/retrain_orchestrator.py` — champion/challenger, promote có margin, rollback, exit codes cho cron |

Điểm **không đổi**: khái niệm nhãn `argmax` cứng (như bạn xác nhận) — báo cáo có thêm
`mean_confidence` để theo dõi độ tự tin trung bình của các nhãn cứng đó.

## 2. Cấu trúc & cách drop-in vào repo

```
fse-training/
├── config/train_config.yaml        → config/train_config.yaml      (MỚI — xóa hyperparameters.yaml)
├── src/training/train_fse_sft.py   → src/training/train_fse_sft.py (GHI ĐÈ)
├── src/training/losses.py          → src/training/losses.py        (GHI ĐÈ)
├── src/training/seed.py            → src/training/seed.py          (MỚI)
├── src/training/evaluate.py        → src/training/evaluate.py      (MỚI)
├── src/data/dataset.py             → src/data/dataset.py           (GHI ĐÈ)
├── scripts/retrain_orchestrator.py → scripts/retrain_orchestrator.py (MỚI)
├── tests/smoke_test.py             → tests/smoke_test.py           (MỚI — kiểm tra pipeline)
├── tests/smoke_model.py            → tests/smoke_model.py          (MỚI — model mini cho test)
└── requirements-training.txt       → gộp vào requirements.txt
```

Sao chép từng file vào đúng vị trí trong repo gốc (giữ nguyên `__init__.py` có sẵn).

## 3. Cài đặt

```bash
pip install -r requirements-training.txt   # mlflow, underthesea, pyarrow, tqdm
# Kiểm tra nhanh pipeline (vài chục giây, không cần GPU / PhoBERT):
python -m tests.smoke_test
```

## 4. Quickstart

```bash
# (a) Chỉ huấn luyện (3 seeds [42, 7, 2026], chọn model tốt nhất, eval test):
python -m src.training.train_fse_sft --config config/train_config.yaml

# (b) Quy trình TỰ ĐỘNG trọn vẹn — cách dùng chính:
python -m scripts.retrain_orchestrator --config config/train_config.yaml --force   # lần đầu
python -m scripts.retrain_orchestrator --config config/train_config.yaml           # các lần sau

# Lệnh tiện ích:
python -m scripts.retrain_orchestrator --dry-run      # chỉ pre-flight, không train
python -m scripts.retrain_orchestrator --skip-train   # promote thủ công từ báo cáo có sẵn
python -m scripts.retrain_orchestrator --rollback     # khôi phục champion trước đó
```

Kết quả sau khi chạy:

```
checkpoints/fse_sft/
├── seed42/  best.pt + meta.json + metrics.jsonl     # mỗi seed một thư mục
├── seed7/   ...
├── seed2026/...
├── selected/  best.pt (seed tốt nhất) + eval_report.json/.md  ← test set
└── training_summary.json                            # F1 theo từng seed, mean ± std
checkpoints/production/
├── best.pt        ← checkpoint đang "gác đếm" phục vụ API
├── manifest.json  ← ai promote, khi nào, F1 bao nhiêu, data_hash, git_hash
└── history/       ← các champion cũ (giữ 3 bản) để rollback
```

## 5. Cấu hình quan trọng (`config/train_config.yaml`)

| Key | Ý nghĩa |
|---|---|
| `segmentation.mode` | `auto` (khuyến nghị): tự phát hiện + tự tách từ nếu thiếu; `off` nếu data đã pre-segmented |
| `data.date_col` | **Đặt `"date"` nếu parquet có cột ngày** → bật kiểm tra split theo thời gian (bỏ trống sẽ nhận cảnh báo) |
| `seeds` | Mặc định `[42, 7, 2026]`; override tạm: `--seeds 42,7` |
| `training.class_weights` | `auto` (khuyến nghị) hoặc ghi đè `[2.75, 0.5, 1.54]` |
| `training.label_smoothing` | 0.05; **không** dùng đồng thời `focal_loss: true` |
| `training.early_stopping.patience` | 3 epoch không cải thiện thì dừng |
| `promote.margin` | Challenger phải hơn champion ít nhất 0.01 F1 mới được thay |
| `model.factory` | Cắm kiến trúc tùy chỉnh, vd `"src.models.fse_model:build_model"` (hàm nhận `model_cfg` trả `nn.Module`); `null` = `AutoModelForSequenceClassification` chuẩn |

> Lưu ý kiến trúc: bản cũ dùng PhoBERT + AttentionPooling tùy chỉnh. Bộ này mặc định
> dùng head phân loại chuẩn của HuggingFace. Muốn giữ AttentionPooling, trỏ
> `model.factory` về class model cũ của bạn — mọi phần còn lại (seed, AMP, loss,
> reporting, orchestrator) hoạt động không đổi.

## 6. Tự động hóa định kỳ

### Cron (retrain hằng tháng, sáng thứ Bảy 06:30 — sau khi daily pipeline đã gom đủ tin tuần)
```bash
crontab -e
# Thêm dòng (chỉnh /path/to/repo cho đúng):
30 6 1-7 * 6 cd /path/to/repo && /usr/bin/python3 -m scripts.retrain_orchestrator \
    --config config/train_config.yaml >> logs/cron_retrain.log 2>&1
```

### GitHub Actions (`.github/workflows/retrain.yml`)
```yaml
name: monthly-retrain
on:
  schedule:
    - cron: "30 23 1 * *"   # 06:30 giờ VN ngày 1 hằng tháng
  workflow_dispatch:         # cho phép bấm chạy tay
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Cài dependencies
        run: pip install -r requirements.txt -r requirements-training.txt
      - name: Smoke test trước khi train
        run: python -m tests.smoke_test
      - name: Retrain + champion/challenger
        run: python -m scripts.retrain_orchestrator --config config/train_config.yaml
      - name: Lưu báo cáo
        if: always()
        uses: actions/upload-artifact@v4
        with: { name: training-reports, path: reports/ }
```

### Ý nghĩa mã thoát (để cron/CI cảnh báo)
| Mã | Nghĩa |
|---|---|
| 0 | Hoàn tất (đã promote HOẶC giữ champion) |
| 1 | Training thất bại — champion giữ nguyên |
| 2 | Pre-flight thất bại (thiếu dữ liệu/config) — không tốn giờ GPU |
| 3 | Lỗi khác (đọc báo cáo, rollback không có history...) |

## 7. Champion/challenger vận hành thế nào?

1. Mỗi lần chạy, challenger (model mới) được đánh giá **trên test set** và sinh `eval_report.json`.
2. So với `checkpoints/production/manifest.json` (champion đang phục vụ API):
   - Tốt hơn **+0.01 F1-macro** (margin chống "thăng chức do nhiễu seed") → **PROMOTE**.
   - Không tốt hơn → **giữ champion**, ghi lý do vào `reports/training/challenger_rejected_*.json`.
3. Promote luôn lưu champion cũ vào `history/` → `--rollback` hoàn tác bất cứ lúc nào.
4. API serving nên đọc `checkpoints/production/best.pt` (hàm
   `src.training.evaluate.load_checkpoint` rebuild model từ metadata — tương thích
   với cả `model.factory` tùy chỉnh).

Xem thí nghiệm: `mlflow ui --backend-store-uri ./mlruns` → mở `http://localhost:5000`.

## 8. Gotchas đã được xử lý sẵn (đừng "sửa ngược")

- **Đừng xóa bước segmentation**: nếu thấy log "Phát hiện dữ liệu ĐÃ tách từ sẵn" là
  data nguồn đã pre-segmented; ngược lại pipeline tự tách + cache vào
  `data/cache/segmentation/`. Cache này được commit-safe (chỉ file JSON hash map).
- **AMP chỉ bật trên GPU** — trên CPU log sẽ báo "AMP tắt", không phải lỗi.
- **MLflow là tùy chọn**: thiếu mlflow pipeline vẫn chạy, chỉ mất tracking.
- **Cảnh báo SPLIT không làm pipeline chết** (trừ thiếu file dữ liệu) — nhưng hãy
  đọc kỹ: đặc biệt là cảnh báo leakage thời gian và trùng lặp chéo train/test.
- Kiểm chứng toàn bộ pipeline trong ~1 phút: `python -m tests.smoke_test`
  (tự tạo dữ liệu giả + model mini trong thư mục tạm, không đụng dữ liệu thật).
