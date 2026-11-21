# Integration Notes — Drop vào repo gốc

## Cách merge file vào repo FSE Engine

Repo gốc có cấu trúc `src/analysis/correlation_stats.py` (script `correlation_robust.py`
import từ đó). Gói fix này bổ sung 6 module mới vào cùng thư mục `src/analysis/`.

### Bước 1: Copy module vào `src/analysis/`

```bash
cp fse-fix-pack/src/analysis/__init__.py        src/analysis/__init__.py.bak  # backup nếu đã có
cp fse-fix-pack/src/analysis/bootstrap.py       src/analysis/
cp fse-fix-pack/src/analysis/breaks.py          src/analysis/
cp fse-fix-pack/src/analysis/causality.py       src/analysis/
cp fse-fix-pack/src/analysis/irf.py             src/analysis/
cp fse-fix-pack/src/analysis/garch_x.py         src/analysis/
cp fse-fix-pack/src/analysis/partial_corr.py    src/analysis/
```

**Nếu `src/analysis/__init__.py` đã có:** merge nội dung bằng tay, giữ export cũ +
thêm 6 module mới. File `__init__.py` của fix pack chỉ re-export 6 module mới.

### Bước 2: Copy 9 script vào `scripts/`

```bash
cp fse-fix-pack/scripts/01_evaluate_model.py        scripts/
cp fse-fix-pack/scripts/02_compare_sentiments.py    scripts/
cp fse-fix-pack/scripts/03_robustness_battery.py    scripts/
cp fse-fix-pack/scripts/04_granger_causality.py     scripts/
cp fse-fix-pack/scripts/05_var_irf.py               scripts/
cp fse-fix-pack/scripts/06_egarch_x.py              scripts/
cp fse-fix-pack/scripts/07_partial_correlation.py   scripts/
cp fse-fix-pack/scripts/08_event_study.py           scripts/
cp fse-fix-pack/scripts/09_forecast_comparison.py   scripts/
```

**Lưu ý về cách chạy:** vì tên file bắt đầu bằng số, KHÔNG chạy được bằng
`python -m scripts.01_evaluate_model` (module name không thể bắt đầu bằng số).
Phải chạy bằng path:

```bash
python scripts/01_evaluate_model.py --test data/processed/test_3class.parquet
```

Toàn bộ script đã có `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
đảm bảo `from src.analysis.X import Y` hoạt động khi chạy theo path.

### Bước 3: Copy config

```bash
cp fse-fix-pack/config/macro_controls.yaml   config/
cp fse-fix-pack/config/events.yaml           config/
```

Nếu repo đã có thư mục `config/`, chỉ merge 2 file YAML này vào.

### Bước 4: Copy tài liệu

```bash
cp fse-fix-pack/docs/depth_extensions.md     docs/
```

### Bước 5: Update `requirements-*.txt`

Thêm các package trong `requirements-additional.txt` vào `requirements-api.txt`
(hoặc tạo `requirements-analysis.txt` mới nếu muốn tách biệt).

```bash
cat fse-fix-pack/requirements-additional.txt >> requirements-api.txt
pip install -r requirements-api.txt
```

## Input cần chuẩn bị trước khi chạy

| Script | Input cần có | Nguồn |
|---|---|---|
| 01 | `data/processed/test_3class.parquet` | do `scripts/prepare_3class_dataset.py` tạo |
| 02 | `data/processed/news_with_probs.parquet` | **CHƯA CÓ** — cần chạy inference PhoBERT trên toàn bộ corpus lịch sử rồi aggregate theo ngày |
| 02 | `data/processed/banking_with_sentiment_vietstock.parquet` | đã có (do `k4_refresh_vietstock.py` tạo) |
| 03 | `data/processed/banking_with_sentiment_vietstock.parquet` | đã có |
| 04 | tương tự 03 | đã có |
| 05 | tương tự 03 | đã có |
| 06 | tương tự 03 | đã có |
| 07 | `data/processed/macro.parquet` (cột: date, vnindex_close, foreign_net_buy_banking, vix, dxy_close, usd_vnd, vnindex_volume) | **CHƯA CÓ** — cần `scripts/fetch_macro.py` hoặc tự điền |
| 08 | `data/processed/vnindex.parquet` (cột: date, vnindex_close) | **CHƯA CÓ** — có thể sinh từ basket banking (script tự fallback) |
| 09 | input chính + cột `s_model` (từ script 02) | phụ thuộc 02 |

### Sinh `news_with_probs.parquet` (bắt buộc cho script 02)

```python
# pseudo-code, chạy bằng notebook hoặc script riêng
import pandas as pd, numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("path/to/phobert_fse")
tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model.eval()

news = pd.read_parquet("data/processed/news_history.parquet")
probs = []
for batch in chunks(news["text"].tolist(), 32):
    inputs = tok(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs.append(torch.softmax(logits, dim=-1).numpy())
news["probs"] = np.concatenate(probs).tolist()
news.to_parquet("data/processed/news_with_probs.parquet")
```

### Sinh `macro.parquet` (bắt buộc cho script 07)

3 biến required: `vnindex_close`, `foreign_net_buy_banking`, `vnindex_volume`.
Các biến khác (vix, dxy_close, usd_vnd) optional — nếu thiếu, partial corr vẫn chạy
nhưng chỉ controls required.

```python
# pseudo-code
import pandas as pd
# VNINDEX: từ scripts/fetch_price_vietstock.py (đã có)
# Foreign net buying: từ data/processed/foreign_flows.parquet nếu đã crawl
macro = pd.DataFrame({
    "date": trading_days,
    "vnindex_close": vnindex_close_series,
    "vnindex_volume": vnindex_volume_series,
    "foreign_net_buy_banking": foreign_net_buy_banking_series,
})
macro.to_parquet("data/processed/macro.parquet")
```

## Lệnh chạy nhanh

```bash
# Từ repo root:
bash run_all.sh    # hoặc chạy từng script riêng:
python scripts/01_evaluate_model.py
python scripts/02_compare_sentiments.py
# ... etc
```

## Xung đột có thể gặp

1. **`src/analysis/__init__.py` đã có** → merge tay, không ghi đè.
2. **`config/` đã có file cùng tên** → backup file cũ rồi copy file mới.
3. **`scikit-learn` cũ** → cần ≥1.3 cho `zero_division` param trong `f1_score`.
4. **`arch` chưa cài** → script 06 sẽ raise `ImportError` với hướng dẫn `pip install arch`.
5. **Tên file bắt đầu bằng số** → không chạy được `-m scripts.01_...`, dùng path.

## Audit trail

Mỗi script ghi `*_summary.json` với trường `interpretation` tiếng Việt — đưa thẳng
vào báo cáo nghiên cứu như một cách trình bày phương pháp.
