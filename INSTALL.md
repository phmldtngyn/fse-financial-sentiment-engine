# INSTALL — đặt bộ kit vào fse-financial-sentiment-engine

Zip có bố cục TRÙNG với repo gốc (file nằm sẵn đúng vị trí). Giải nén THẲNG vào
folder gốc repo (nơi có `docker-compose.yml`). Chọn đúng hệ điều hành:

## Windows PowerShell (5.1 trở lên)

```powershell
# Buoc 1: giai nen THANG vao folder goc repo (chu y: khong cd di dau)
Expand-Archive -Path .\fse-kits.zip -DestinationPath . -Force

# Buoc 2: ra lai thay doi
git status --short

# Buoc 3: phu thuoc + test (chay trong venv)
pip install -r requirements-ops.txt
python -m pytest tests/test_ops_units.py tests/test_cli_flows.py -q
python -m tests.smoke_test

# Buoc 4: env + /health (2 viec thu cong duy nhat)
Copy-Item .env.example .env
#   them vao api/main.py:
#   @app.get("/health")
#   async def health(): return {"status": "ok"}

# Buoc 5: container (PS 5.1 khong co "&&" -> tach tung lenh)
docker compose config -q
docker compose up -d
docker compose ps

# Buoc 6: van hanh dau tien (xem docs/kits/README-ops.md muc 2)
python -m scripts.prepare_splits --source data/raw/vn_quant_fulltext.parquet --score-col Sentiment_Score --date-col date
python -m scripts.init_db --db data/fse.db --migrate
python -m scripts.apply_news_alignment --input data/parsed/articles.parquet --to-sqlite data/fse.db
```

## Linux / macOS

```bash
unzip -o fse-kits.zip -d .
git status --short
pip install -r requirements-ops.txt
python -m pytest tests/test_ops_units.py tests/test_cli_flows.py -q   # 33 passed
python -m tests.smoke_test                                            # 25/25
cp .env.example .env
# thêm @app.get("/health") vào api/main.py
docker compose config -q && docker compose up -d
docker compose ps
python -m scripts.prepare_splits --source data/raw/vn_quant_fulltext.parquet --score-col Sentiment_Score --date-col date
python -m scripts.init_db --db data/fse.db --migrate
python -m scripts.apply_news_alignment --input data/parsed/articles.parquet --to-sqlite data/fse.db
```

## Zip này CỐ TÌNH không chứa

- `__init__.py` của `src/`, `src/data/`, `src/training/`, `src/storage/` — giữ nguyên
  bản của repo (code kit import theo kiểu module, không cần re-export).
- Chỉ THÊM MỚI: `scripts/__init__.py`, `src/analysis/__init__.py`, `tests/__init__.py`
  (repo chưa có — cần để `python -m scripts.xxx` / `python -m tests.xxx` chạy chắc chắn).

## Dọn dẹp khuyến nghị (sau khi mọi thứ chạy ổn)

```bash
# RAG đã NGỪNG (quyết định audit): compose mới đã bỏ mount data/rag_index.
# Dọn hẳn — xóa route /rag trong api/main.py TRƯỚC rồi mới chuyển code:
git mv src/rag archive/rag_src
git mv scripts/build_rag_index.py scripts/test_rag.py archive/

# Dummy data — khi đã có dữ liệu thật:
git mv scripts/create_dummy_gold.py scripts/create_dummy_market.py archive/
```
