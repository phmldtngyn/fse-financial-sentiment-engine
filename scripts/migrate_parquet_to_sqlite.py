"""
Migrate dữ liệu parquet hiện có vào SQLite — chạy 1 lần khi chuyển tầng lưu trữ.

Chạy từ gốc repo:
    python -m scripts.migrate_parquet_to_sqlite \
        --db data/fse.db \
        --market data/processed/banking_daily.parquet \
        --sentiment data/processed/banking_with_sentiment.parquet

Cột mặc định kỳ vọng (chỉnh qua CLI nếu tên khác):
  market:    ticker, date, [open, high, low, close, volume]
  sentiment: ticker, date, sentiment  → sentiment_daily.s_t
  articles:  text, [url, source, title, published_at, trading_date, tickers]
Idempotent: chạy lại chỉ REPLACE, không nhân đôi.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.storage.sqlite_db import connect, finish_run, log_run, \
    upsert_articles, upsert_market_daily, upsert_sentiment_daily

logger = logging.getLogger("fse.migrate")


def load_mapped(path: Path, colmap: dict[str, str]) -> pd.DataFrame | None:
    if not path.exists():
        logger.info("Bỏ qua %s (không tồn tại)", path)
        return None
    df = pd.read_parquet(path)
    missing = [dst for dst, src in colmap.items() if src not in df.columns]
    if missing:
        logger.warning("Bỏ qua %s: thiếu cột %s (có: %s) — chỉnh CLI cho khớp.",
                       path, missing, list(df.columns))
        return None
    out = pd.DataFrame({dst: df[src] for dst, src in colmap.items()})
    logger.info("Đọc %s: %d dòng", path.name, len(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate parquet → SQLite")
    parser.add_argument("--db", default="data/fse.db")
    parser.add_argument("--market", default="data/processed/banking_daily.parquet")
    parser.add_argument("--sentiment", default="data/processed/banking_with_sentiment.parquet")
    parser.add_argument("--articles", default=None,
                        help="Parquet tin đã align (articles_aligned.parquet)")
    parser.add_argument("--sent-col", default="sentiment")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    market = load_mapped(Path(args.market), {
        "ticker": "ticker", "date": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume"})
    sentiment = load_mapped(Path(args.sentiment), {
        "ticker": "ticker", "trading_date": "date", "s_t": args.sent_col,
        "n_articles": "n_articles", "mean_conf": "mean_confidence"})
    articles = None
    if args.articles:
        articles = load_mapped(Path(args.articles), {
            "text": "text", "url": "url", "source": "source", "title": "title",
            "published_at": "published_at", "trading_date": "trading_date",
            "tickers": "tickers", "is_duplicate": "is_duplicate"})
        if articles is not None:
            # tickers đang là chuỗi JSON trong parquet → giữ nguyên chuỗi là hợp lệ
            articles["tickers"] = articles["tickers"].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else (v or []))

    with connect(args.db) as conn:
        run_id = log_run(conn, "migrate_parquet_to_sqlite")
        counts = {}
        try:
            if market is not None:
                counts["market_daily"] = upsert_market_daily(conn, market)
            if sentiment is not None:
                counts["sentiment_daily"] = upsert_sentiment_daily(conn, sentiment)
            if articles is not None:
                counts["articles"] = upsert_articles(conn, articles)
            finish_run(conn, run_id, "ok", json.dumps(counts))
        except Exception as exc:
            finish_run(conn, run_id, "failed", str(exc))
            raise

    if not counts:
        logger.warning("Không migrate được bảng nào — kiểm tra tên cột qua --help.")
        return 2
    logger.info("Migrate xong: %s → %s", counts, args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
