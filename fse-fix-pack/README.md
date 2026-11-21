# FSE Engine — Depth Extension Pack (Fix Pack)

Gói mở rộng chiều sâu cho bài nghiên cứu FSE (Financial Sentiment Engine),
chuyển từ "khảo sát tương quan" sang "kiểm định nhân quả + giá trị thông tin gia tăng".

## Nội dung gói

```
fse-fix-pack/
├── README.md                       ← file này
├── INTEGRATION_NOTES.md            ← hướng dẫn tích hợp vào repo gốc
├── requirements-additional.txt     ← phụ thuộc Python thêm vào
├── run_all.sh                      ← chạy tuần tự toàn bộ 9 script
├── config/
│   ├── macro_controls.yaml         ← cấu hình biến macro cho partial corr
│   └── events.yaml                 ← cấu hình 7 event study
├── src/analysis/
│   ├── __init__.py
│   ├── bootstrap.py                ← block bootstrap CI cho tương quan
│   ├── breaks.py                   ← Bai–Perron + CUSUM structural breaks
│   ├── causality.py                ← Granger + Toda–Yamamoto
│   ├── irf.py                      ← Local Projections (Jordà 2005)
│   ├── garch_x.py                  ← EGARCH-X / GJR-GARCH-X
│   └── partial_corr.py             ← partial corr + incremental R²
├── scripts/
│   ├── 01_evaluate_model.py        ← F1, confusion matrix, kappa
│   ├── 02_compare_sentiments.py    ← PhoBERT vs Gemini S_t
│   ├── 03_robustness_battery.py    ← bootstrap CI + subsample + breaks
│   ├── 04_granger_causality.py     ← Granger + Toda–Yamamoto 2 hướng
│   ├── 05_var_irf.py               ← Local Projections + VAR Cholesky
│   ├── 06_egarch_x.py              ← EGARCH-X + GJR-GARCH-X
│   ├── 07_partial_correlation.py   ← partial corr với macro controls
│   ├── 08_event_study.py           ← CAR quanh 7 events
│   └── 09_forecast_comparison.py   ← walk-forward + DM/CW tests
└── docs/
    └── depth_extensions.md         ← phương pháp luận + citations
```

## Tổng quan 9 script

| # | Script | Trả lời câu hỏi | Tier |
|---|---|---|---|
| 01 | evaluate_model | F1-macro thật của PhoBERT là bao nhiêu? (lấp placeholder 0.XX) | 1 — bắt buộc |
| 02 | compare_sentiments | PhoBERT và Gemini có tương quan không? Có cùng dự báo tương quan không? | 1 — bắt buộc |
| 03 | robustness_battery | Bảng 8 cặp có ổn định qua subsample và bootstrap CI không? | 1 — bắt buộc |
| 04 | granger_causality | Sentiment Granger-cause return không? Cả 2 hướng? | 2 — nhân quả |
| 05 | var_irf | Độ lớn và độ bền tác động qua IRF 14 ngày? | 2 — nhân quả |
| 06 | egarch_x | Sentiment có vào mean/variance equation không? Leverage? | 3 — vol |
| 07 | partial_correlation | Sentiment còn significant sau khi partial out macro không? | 5 — quan trọng nhất |
| 08 | event_study | Sentiment trước event có tiên đoán CAR sau event không? | 4 — event |
| 09 | forecast_comparison | Sentiment có tăng RMSE/MAE out-of-sample không? DM test? | 6 — predictive |

## Cài đặt

```bash
# 1. Copy file vào repo (xem INTEGRATION_NOTES.md)
cd /path/to/fse-financial-sentiment-engine
cp -r /path/to/fse-fix-pack/src/analysis/*.py src/analysis/
cp -r /path/to/fse-fix-pack/scripts/*.py scripts/
cp -r /path/to/fse-fix-pack/config/ config/  # merge vào nếu đã có config/
cp -r /path/to/fse-fix-pack/docs/depth_extensions.md docs/

# 2. Cài thêm phụ thuộc
pip install -r requirements-additional.txt

# 3. Chạy tuần tự (tốn ~30–60 phút tuỳ corpus)
bash run_all.sh
```

## Thứ tự chạy

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
              (03-09 có thể chạy song song nếu đủ tài nguyên)
```

**Lý do thứ tự:**
- `01` phải chạy trước để có F1 thật, và để sinh probs cho `02`
- `02` sinh `s_model` cột (S_t từ PhoBERT), dùng ở `09`
- `03` độc lập nhưng nên chạy sau khi đã có bảng tương quan chính
- `04`–`09` độc lập với nhau

## Kết quả đầu ra

Mỗi script ghi vào `data/processed/<module>/`:
- CSV / parquet: bảng số liệu đầy đủ
- JSON: summary + diễn giải
- PNG: biểu đồ (confusion matrix, IRF, scatter)

Cuối cùng, đọc các `*_summary.json` để có số liệu đưa vào báo cáo.

## Phụ thuộc

| Package | Phiên bản | Dùng cho |
|---|---|---|
| `arch` | ≥6.1 | EGARCH-X, GJR-GARCH-X |
| `statsmodels` | ≥0.14 | VAR + Cholesky IRF |
| `scikit-learn` | ≥1.3 | F1, confusion matrix |
| `pyyaml` | ≥6.0 | đọc config YAML |
| `matplotlib` | ≥3.8 | plot |
| `scipy` | ≥1.11 | kiểm định thống kê |

(các package `pandas`, `numpy` đã có sẵn trong FSE Engine)
