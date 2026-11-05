"""
Tầng lưu trữ SQLite cho pipeline chạy liên tục (WAL mode, upsert idempotent).

Lựa chọn thiết kế (theo quyết định "SQLite bây giờ"):
- 1 process ghi mỗi ngày (daily pipeline) + API/ dashboard đọc → SQLite WAL là đủ,
  không cần vận hành Postgres. Khi cần swap Postgres: chỉ phần này thay đổi,
  các hàm upsert/get giữ nguyên interface.

Idempotency: mọi bảng có khóa tự nhiên (vd articles.id = md5(url+text),
sentiment_daily PK (ticker, trading_date)) → chạy lại ngày hôm đó không nhân đôi.

Sử dụng:
    from src.storage.sqlite_db import connect, upsert_articles, get_sentiment
    with connect("data/fse.db") as conn:
        upsert_articles(conn, df_articles)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,          -- md5(url | text)
    url           TEXT,
    source        TEXT,
    title         TEXT,
    text          TEXT NOT NULL,
    published_at  TEXT,                      -- ISO 8601 giờ đăng thật
    trading_date  TEXT,                      -- phiên gán (sau apply_news_alignment)
    tickers       TEXT,                      -- JSON list, vd ["VCB","BID"]
    is_duplicate  INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_trading_date ON articles(trading_date);
CREATE INDEX IF NOT EXISTS idx_articles_ticker ON articles(tickers);

CREATE TABLE IF NOT EXISTS sentiment_daily (
    ticker        TEXT NOT NULL,
    trading_date  TEXT NOT NULL,
    s_t           REAL,                      -- chỉ số sentiment tổng hợp trong ngày
    n_articles    INTEGER,
    mean_conf     REAL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (ticker, trading_date)
);

CREATE TABLE IF NOT EXISTS market_daily (
    ticker        TEXT NOT NULL,
    date          TEXT NOT NULL,
    open          REAL, high REAL, low REAL, close REAL,
    volume        REAL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job         TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',   -- running|ok|failed
    detail      TEXT
);
"""


def article_id(url: str | None, text: str) -> str:
    """ID ổn định của bài báo — chìa khóa idempotency cho crawl chạy lại."""
    return hashlib.md5(f"{url or ''}|{text}".encode("utf-8")).hexdigest()


@contextmanager
def connect(db_path: str | Path = "data/fse.db"):
    """Mở kết nối WAL + auto-init schema. `with connect(...) as conn:`"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_articles(conn: sqlite3.Connection, df) -> int:
    """Ghi/bài báo (INSERT OR REPLACE theo id). df cần: text, các cột khác tùy chọn."""
    import pandas as pd

    rows = []
    for _, r in df.iterrows():
        rows.append((
            article_id(r.get("url"), str(r["text"])),
            r.get("url"), r.get("source"), r.get("title"), str(r["text"]),
            str(r.get("published_at") or ""),
            str(r.get("trading_date") or ""),
            json.dumps(r.get("tickers") or [], ensure_ascii=False),
            int(bool(r.get("is_duplicate", False))),
            _now(),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO articles (id,url,source,title,text,published_at,"
        "trading_date,tickers,is_duplicate,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def upsert_sentiment_daily(conn: sqlite3.Connection, df) -> int:
    """df cần: ticker, trading_date, s_t; tùy chọn n_articles, mean_conf."""
    rows = [(
        r["ticker"], str(r["trading_date"]),
        None if r.get("s_t") is None or (isinstance(r.get("s_t"), float) and r.get("s_t") != r.get("s_t")) else float(r["s_t"]),
        int(r["n_articles"]) if r.get("n_articles") is not None else None,
        float(r["mean_conf"]) if r.get("mean_conf") is not None else None,
        _now(),
    ) for _, r in df.iterrows()]
    conn.executemany(
        "INSERT OR REPLACE INTO sentiment_daily (ticker,trading_date,s_t,n_articles,"
        "mean_conf,updated_at) VALUES (?,?,?,?,?,?)", rows)
    return len(rows)


def upsert_market_daily(conn: sqlite3.Connection, df) -> int:
    """df cần: ticker, date, close; tùy chọn open/high/low/volume."""
    rows = [(
        r["ticker"], str(r["date"]),
        float(r["open"]) if r.get("open") is not None else None,
        float(r["high"]) if r.get("high") is not None else None,
        float(r["low"]) if r.get("low") is not None else None,
        float(r["close"]) if r.get("close") is not None else None,
        float(r["volume"]) if r.get("volume") is not None else None,
        _now(),
    ) for _, r in df.iterrows()]
    conn.executemany(
        "INSERT OR REPLACE INTO market_daily (ticker,date,open,high,low,close,volume,"
        "updated_at) VALUES (?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def log_run(conn: sqlite3.Connection, job: str, status: str = "running",
            detail: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO pipeline_runs (job, started_at, status, detail) VALUES (?,?,?,?)",
        (job, _now(), status, detail))
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, status: str = "ok",
               detail: str | None = None) -> None:
    conn.execute("UPDATE pipeline_runs SET finished_at=?, status=?, detail=? WHERE id=?",
                 (_now(), status, detail, run_id))


def get_sentiment(conn: sqlite3.Connection, ticker: str | None = None,
                  start: str | None = None, end: str | None = None):
    """Đọc sentiment_daily, lọc theo ticker và khoảng ngày (INCLUDE cả 2 đầu)."""
    q, args = "SELECT * FROM sentiment_daily WHERE 1=1", []
    if ticker:
        q += " AND ticker=?"; args.append(ticker)
    if start:
        q += " AND trading_date>=?"; args.append(start)
    if end:
        q += " AND trading_date<=?"; args.append(end)
    return pd_read(conn, q, args)


def get_market(conn: sqlite3.Connection, ticker: str | None = None,
               start: str | None = None, end: str | None = None):
    q, args = "SELECT * FROM market_daily WHERE 1=1", []
    if ticker:
        q += " AND ticker=?"; args.append(ticker)
    if start:
        q += " AND date>=?"; args.append(start)
    if end:
        q += " AND date<=?"; args.append(end)
    return pd_read(conn, q, args)


def pd_read(conn: sqlite3.Connection, query: str, args=None):
    import pandas as pd

    return pd.read_sql_query(query, conn, params=args or [])
