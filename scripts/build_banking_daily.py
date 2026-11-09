# -*- coding: utf-8 -*-
"""
build_banking_daily.py
======================
Tong hop gia 26 ma ngan hang (tu fetch_price_vietstock.py) thanh
bang daily cap NGANH, schema giong het banking_daily.parquet
do scripts/extract_banking_daily.py sinh ra tu DB goc:

  trade_date, total_volume, avg_return, avg_range, avg_close, n_banks

Dinh nghia (khop extract_banking_daily.py):
  intraday_range = (high - low) / close     (tung ma)
  total_volume   = SUM(volume)              (ca nhom theo ngay)
  avg_return     = MEAN(pct)                (pct = close/prev_close - 1)
  avg_range      = MEAN(intraday_range)
  avg_close      = MEAN(close)
  n_banks        = so ma co giao dich trong ngay
  loc: chi giu ngay co n_banks >= 5

Cach chay:
  python scripts/build_banking_daily.py
  python scripts/build_banking_daily.py --in data/processed/vietstock_price_daily.parquet
  python scripts/build_banking_daily.py --csv-out data/processed/banking_daily_vietstock.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="data/processed/vietstock_price_daily.parquet",
                    help="Parquet gia tung ma (tu fetch_price_vietstock.py).")
    ap.add_argument("--out",
                    default="data/processed/banking_daily_vietstock.parquet",
                    help="Parquet daily nganh dau ra.")
    ap.add_argument("--csv-out",
                    default="data/processed/banking_daily_vietstock.csv",
                    help="CSV de xem Excel (khong ghi neu de rong).")
    ap.add_argument("--min-banks", type=int, default=5,
                    help="So ma toi thieu trong ngay (mac dinh 5).")
    args = ap.parse_args()

    print("=" * 64)
    print("BUILD BANKING DAILY (Vietstock) -> khop schema DB goc")
    print("=" * 64)
    df = pd.read_parquet(args.inp)
    print(f"Input : {args.inp}  ({len(df):,} dong, "
          f"{df['ticker'].nunique()} ma)")

    need = {"ticker", "date", "high", "low", "close", "volume", "pct"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Thieu cot: {missing} - kiem tra file input.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["volume"].notna()) & (df["volume"] > 0) &
            (df["close"].notna())]
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"]

    agg = (df.groupby("date")
             .agg(total_volume=("volume", "sum"),
                  avg_return=("pct", "mean"),
                  avg_range=("intraday_range", "mean"),
                  avg_close=("close", "mean"),
                  n_banks=("ticker", "nunique"))
             .reset_index()
             .rename(columns={"date": "trade_date"}))
    agg = agg[agg["n_banks"] >= args.min_banks].reset_index(drop=True)
    agg = agg.sort_values("trade_date").reset_index(drop=True)

    print(f"\nNgay  : {agg['trade_date'].min()} -> {agg['trade_date'].max()}"
          f"  ({len(agg):,} ngay giao dich)")
    print("\nStats:")
    print(agg[["total_volume", "avg_return", "avg_range",
               "n_banks"]].describe().round(4).to_string())
    print("\n5 ngay gan nhat:")
    print(agg.tail(5).to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.out, index=False)
    print(f"\nOK -> {args.out}")
    if args.csv_out:
        agg.to_csv(args.csv_out, index=False, encoding="utf-8-sig")
        print(f"OK -> {args.csv_out}")

    print("\nBuoc tiep (mau sua join_banking_sentiment.py):")
    print('  BANKING_PATH = Path("data/processed/'
          'banking_daily_vietstock.parquet")')


if __name__ == "__main__":
    main()
