"""
Cắn tin thô → bài báo "sạch" sẵn sàng cho suy luận sentiment hằng ngày.

Ghép 3 mảnh từng thiếu trong repo:
1. Ticker mapping: tên công ty/bí danh → mã CK (config/ticker_map.yaml)
2. Trading session: published_at → ngày phiên giao dịch (tin sau 15:00/CN/lễ → phiên kế)
3. Dedup chéo nguồn: vân tay tiêu đề + fuzzy 85%

Input: parquet tin thô từ crawler (mặc định data/parsed/articles.parquet),
các cột tùy chỉnh qua CLI. Output: data/processed/articles_aligned.parquet
(id ổn định, tickers JSON, trading_date, is_duplicate) — có thể upsert thẳng
vào SQLite qua scripts.init_db / sqlite_db.upsert_articles.

Chạy từ gốc repo:
    python -m scripts.apply_news_alignment --input data/parsed/articles.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.dedup import find_duplicates
from src.data.market_calendar import assign_session, load_holidays
from src.data.ticker_mapping import load_ticker_map, match_tickers
from src.storage.sqlite_db import article_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Align tin thô: ticker + phiên + dedup")
    parser.add_argument("--input", default="data/parsed/articles.parquet")
    parser.add_argument("--text-col", default="text", help="Cột nội dung bài (text hoặc title+text)")
    parser.add_argument("--title-col", default=None, help="Cột tiêu đề nếu có (dedup dùng)")
    parser.add_argument("--date-col", default="published_at", help="Cột thời điểm đăng")
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--ticker-map", default="config/ticker_map.yaml")
    parser.add_argument("--holidays", default="config/holidays_vn.yaml")
    parser.add_argument("--out", default="data/processed/articles_aligned.parquet")
    parser.add_argument("--to-sqlite", default=None, metavar="DB",
                        help="Upsert thêm vào SQLite, vd --to-sqlite data/fse.db")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("fse.align")

    path = Path(args.input)
    if not path.exists():
        print(f"[LỖI] Không tìm thấy file tin thô: {path}", file=sys.stderr)
        return 2
    df = pd.read_parquet(path)

    text_col = args.text_col if args.text_col in df.columns else (
        args.title_col if args.title_col in df.columns else None)
    if text_col is None or args.date_col not in df.columns:
        print(f"[LỖI] Thiếu cột cần thiết. Có: {list(df.columns)} "
              f"(cần '{args.text_col}' hoặc '{args.title_col}' + '{args.date_col}')",
              file=sys.stderr)
        return 2

    holidays = load_holidays(args.holidays)
    mapping = load_ticker_map(args.ticker_map)

    title_col = args.title_col if (args.title_col and args.title_col in df.columns) else text_col
    df = df.sort_values(args.date_col).reset_index(drop=True)

    logger.info("Align %d bài từ %s ...", len(df), path.name)
    df["tickers"] = [json.dumps(match_tickers(t, mapping), ensure_ascii=False)
                     for t in df[text_col].astype(str)]
    df["trading_date"] = [assign_session(v, holidays=holidays).isoformat()
                          for v in pd.to_datetime(df[args.date_col])]
    df["id"] = [article_id(r.get(args.url_col), str(r[text_col]))
                for _, r in df.iterrows()]
    df["is_duplicate"] = find_duplicates(
        df, title_col=title_col, date_col=args.date_col, window_days=14).astype(int)

    out_cols = ["id", args.url_col, args.source_col, title_col, text_col,
                args.date_col, "trading_date", "tickers", "is_duplicate"]
    # khử trùng lặp (title_col == text_col khi bài không có cột tiêu đề riêng)
    out_cols = list(dict.fromkeys(c for c in out_cols if c in df.columns))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_parquet(out, index=False)

    n_dup = int(df["is_duplicate"].sum())
    n_no_ticker = int((df["tickers"] == "[]").sum())
    logger.info("Xong: %d bài | trùng lặp: %d (%.1f%%) | không khớp ticker: %d (%.1f%%)",
                len(df), n_dup, 100 * n_dup / max(1, len(df)),
                n_no_ticker, 100 * n_no_ticker / max(1, len(df)))
    logger.info("Phân bố theo phiên (10 ngày gần nhất):\n%s",
                df.groupby("trading_date").size().tail(10).to_string())
    logger.info("→ %s", out)

    if args.to_sqlite:
        from src.storage.sqlite_db import connect, upsert_articles

        with connect(args.to_sqlite) as conn:
            n = upsert_articles(conn, df[out_cols].rename(
                columns={args.url_col: "url", args.source_col: "source",
                         title_col: "title", text_col: "text",
                         args.date_col: "published_at"}))
        logger.info("Upsert %d dòng → %s", n, args.to_sqlite)

    if n_no_ticker / max(1, len(df)) > 0.5:
        logger.warning("Hơn 50%% bài không khớp ticker — bổ sung alias vào "
                       "config/ticker_map.yaml hoặc kiểm tra cột text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
