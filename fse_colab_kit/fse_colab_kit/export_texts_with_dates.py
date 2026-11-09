"""
Phase A - Xuat (text, date) tu master_quant_database.db | BAN VA FONT v2
Chay: python export_texts_with_dates.py
"""
import argparse
import sqlite3
import sys
from pathlib import Path

# [VÁ 1] Ép UTF-8 cho stdout/stderr
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd


def log(*a) -> None:
    """In an toàn: console từ chối ký tự thì in bản escape thay vì crash."""
    try:
        print(*a)
    except UnicodeEncodeError:
        print(*[x.encode("utf-8", "replace").decode("ascii", "backslashreplace") if isinstance(x, str) else x
                for x in a])


def build_text(sentiment: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Join sentiment <-> news qua Title (exact, fallback prefix 100 ký tự) + dựng text."""
    for df in (sentiment, news):
        df["title_key"] = df["Title"].astype(str).str.strip().str.lower()

    merged = sentiment[["Date", "Title", "title_key"]].merge(
        news[["title_key", "Content", "Date"]],
        on="title_key", how="inner", suffixes=("", "_news"),
    )
    method = "exact title"
    merged = merged.drop_duplicates(subset=["title_key"])

    if len(merged) < len(sentiment) * 0.5:
        for df in (sentiment, news):
            df["title_prefix"] = df["title_key"].str[:100]
        merged = sentiment[["Date", "Title", "title_prefix"]].merge(
            news[["title_prefix", "Content", "Date"]],
            on="title_prefix", how="inner", suffixes=("", "_news"),
        )
        merged = merged.drop_duplicates(subset=["title_prefix"])
        method = "title prefix (100 ký tự)"

    log(f"Join qua {method}: {len(merged):,} dòng")

    merged["text"] = (
        merged["Title"].fillna("").astype(str).str.strip()
        + " . "
        + merged["Content"].fillna("").astype(str).str.strip()
    ).str[:800].str.strip()

    before = len(merged)
    merged = merged.dropna(subset=["Content"])
    merged = merged[merged["text"].str.len() >= 50].reset_index(drop=True)
    log(f"Sau khi loc: {len(merged):,} / {before:,} dong")

    return merged[["Date_news", "text"]].rename(columns={"Date_news": "date"})


def main() -> int:
    ap = argparse.ArgumentParser(description="Xuat text + date tu master DB (Phase A)")
    ap.add_argument("--db", default="data/raw/vn_quant/master_quant_database.db")
    ap.add_argument("--out", default="texts_with_dates.csv")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        log(f"[LOI] Khong tim thay DB: {db}")
        log('      Chi dinh: python export_texts_with_dates.py --db "duong-dan-file.db"')
        return 2

    con = sqlite3.connect(db)
    # [VÁ 2] Byte hỏng trong DB -> thay bằng '?' thay vì crash
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    sentiment = pd.read_sql_query("SELECT Date, Title FROM sentiment_LLM_labeled", con)
    news = pd.read_sql_query(
        "SELECT Date, Title, Content FROM google_news_10years_FULLTEXT", con
    )
    con.close()
    log(f"sentiment_LLM_labeled       : {len(sentiment):,} dong")
    log(f"google_news_10years_FULLTEXT: {len(news):,} dong")

    out = build_text(sentiment, news)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)

    # [VÁ 3] utf-8-sig (BOM) -> Excel/PowerShell hien thi dung tieng Viet
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    log("\n" + "=" * 60)
    log(f"DA LUU: {args.out}")
    log(f"  So dong   : {len(out):,} (ky vong ~6.458)")
    log(f"  Date range: {out['date'].min():%Y-%m-%d} -> {out['date'].max():%Y-%m-%d}")
    # [VÁ 4] Dong mau de kiem tra font ngay luc chay
    log("  Dong mau  : " + str(out.iloc[0]["text"])[:80])
    log("=" * 60)
    log("BUOC TIEP: upload texts_with_dates.csv vao Drive /MyDrive/FSE_Project/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())