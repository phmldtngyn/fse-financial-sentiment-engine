"""
Tạo SQLite DB với schema đầy đủ + (tùy chọn) migrate dữ liệu parquet hiện có.

Chạy từ gốc repo:
    python -m scripts.init_db --db data/fse.db                       # chỉ tạo schema
    python -m scripts.init_db --db data/fse.db --migrate             # tạo + migrate nếu có parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.storage.sqlite_db import connect, log_run, finish_run, upsert_articles, \
    upsert_sentiment_daily, upsert_market_daily


def migrate_if_exists(db_path: str, logger) -> None:
    """Migrate các parquet tiêu biểu nếu tồn tại (cột nhận diện theo tên phổ biến)."""
    import pandas as pd

    candidates = {
        "articles": ("data/parsed/articles.parquet",
                     {"text": "text", "url": "url", "source": "source",
                      "title": "title", "published_at": "published_at",
                      "trading_date": "trading_date"}),
        "sentiment": ("data/processed/banking_with_sentiment.parquet",
                      {"ticker": "ticker", "trading_date": "date", "s_t": "sentiment"}),
        "market": ("data/processed/banking_daily.parquet",
                   {"ticker": "ticker", "date": "date", "open": "open", "high": "high",
                    "low": "low", "close": "close", "volume": "volume"}),
    }
    for name, (path_str, cols) in candidates.items():
        path = Path(path_str)
        if not path.exists():
            logger.info("Bỏ qua %s (không có %s)", name, path)
            continue
        df = pd.read_parquet(path)
        missing = [c for c in cols.values() if c not in df.columns and c not in ("url", "source", "title")]
        if missing:
            logger.warning("Bỏ qua %s: thiếu cột %s (có: %s) — chỉnh migrate_if_exists cho khớp.",
                           name, missing, list(df.columns))
            continue
        with connect(db_path) as conn:
            run_id = log_run(conn, f"migrate_{name}")
            if name == "articles":
                n = upsert_articles(conn, df)
            elif name == "sentiment":
                n = upsert_sentiment_daily(conn, df.rename(columns={cols["s_t"]: "s_t"}))
            else:
                n = upsert_market_daily(conn, df)
            finish_run(conn, run_id, "ok", f"{n} rows")
        logger.info("Migrate %s: %d dòng → %s", name, n, db_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/fse.db")
    parser.add_argument("--migrate", action="store_true")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("fse.init_db")

    with connect(args.db) as conn:
        run_id = log_run(conn, "init_db")
        finish_run(conn, run_id, "ok", "schema created")
    logger.info("DB sẵn sàng: %s", args.db)

    if args.migrate:
        migrate_if_exists(args.db, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
