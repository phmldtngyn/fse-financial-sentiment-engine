# -*- coding: utf-8 -*-
"""
fetch_price_vietstock.py
========================
Lay gia sach lenh hang ngay (OHLCV) cho nhom 26 ma ngan hang tu trang
cong khai Vietstock Finance (finance.vietstock.vn/lich-su-gia).

Cach hoat dong (da kiem chinh that, 2026-11):
  1. GET trang  https://finance.vietstock.vn/{TICKER}/lich-su-gia.htm
     -> lay cookie session + __RequestVerificationToken
  2. POST https://finance.vietstock.vn/data/getpricehistory
     body JSON: {"Code": ..., "FromDate": ..., "ToDate": ...,
                 "PageIndex": 1, "PageSize": 2000,
                 "__RequestVerificationToken": ...}
     -> JSON list: TradingDate(epoch ms), OpenPrice, HighestPrice,
        LowestPrice, ClosePrice, TotalVol, TotalVal, Exchange, Rows(total)
  3. Luu tung ma 1 CSV + gop thanh 1 parquet dung chung.

Dau ra khop schema UNIFIED_RESEARCH_MATRIX (cot: ticker, date, open, high,
low, close, volume, pct) de thay truc tiep cho extract_banking_daily.py.
pct = close/prev_close - 1 (phan so, giong DB goc); gia theo nghin dong.

Cach chay:
  python scripts/fetch_price_vietstock.py                        # 26 ma, 2022-01-01 -> hom nay
  python scripts/fetch_price_vietstock.py --start 2020-01-01
  python scripts/fetch_price_vietstock.py --tickers VCB,BID --end 2026-04-30
  python scripts/fetch_price_vietstock.py --skip-csv             # chi ghi parquet

Yeu cau: python 3.9+, requests, pandas, pyarrow
"""
import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------------------------------------------------------
# CAU HINH
# ----------------------------------------------------------------------------
BANKING_TICKERS = [
    "VCB", "BID", "CTG", "TCB", "VPB", "MBB", "ACB", "STB", "SHB", "HDB",
    "LPB", "TPB", "VIB", "MSB", "OCB", "SSB", "EIB", "NAB", "BVB", "SGB",
    "VBB", "KLB", "PGB", "ABB", "BAB", "PVCB",
]

PAGE_URL = "https://finance.vietstock.vn/{code}/lich-su-gia.htm"
API_URL = "https://finance.vietstock.vn/data/getpricehistory"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
VN_TZ = timezone(timedelta(hours=7))
MAX_RETRY = 3
TOKEN_RE = re.compile(
    r'name=__RequestVerificationToken type=hidden value=([A-Za-z0-9_\-]+)')


# ----------------------------------------------------------------------------
# PHIEN VIETSTOCK: cookie + token
# ----------------------------------------------------------------------------
class VietstockSession:
    """Quan ly session Vietstock: cookie + anti-forgery token."""

    def __init__(self, sleep_sec: float = 1.0):
        self.sleep_sec = sleep_sec
        self.http = requests.Session()
        self.http.headers.update({
            "User-Agent": UA,
            "Accept-Language": "vi-VN,vi;q=0.9",
        })
        self._token = None
        self._last_request = 0.0

    # ---- etiquette: gioi han tan so truy cap ----
    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.sleep_sec:
            time.sleep(self.sleep_sec - elapsed)
        self._last_request = time.time()

    def refresh(self, code: str = "VCB"):
        """GET 1 trang lich su gia de nhan cookie + token moi."""
        self._throttle()
        r = self.http.get(PAGE_URL.format(code=code), timeout=30)
        r.raise_for_status()
        m = TOKEN_RE.search(r.text)
        if not m:
            raise RuntimeError(
                "Khong tim thay __RequestVerificationToken trong trang "
                "(Vietstock co the da doi cau truc trang).")
        self._token = m.group(1)

    def price_history(self, code: str, from_d: str, to_d: str,
                      page_size: int = 2000):
        """POST 1 khoang ngay -> list dict dong gia (moi nhat truoc)."""
        if self._token is None:
            self.refresh(code)
        payload = {
            "Code": code.upper(),
            "FromDate": from_d,
            "ToDate": to_d,
            "PageIndex": 1,
            "PageSize": page_size,
            "__RequestVerificationToken": self._token,
        }
        last_err = None
        for attempt in range(1, MAX_RETRY + 1):
            self._throttle()
            try:
                r = self.http.post(
                    API_URL, data=json.dumps(payload), timeout=30,
                    headers={
                        "Content-Type": "application/json; charset=UTF-8",
                        "X-Requested-With": "XMLHttpRequest",
                        "Origin": "https://finance.vietstock.vn",
                        "Referer": PAGE_URL.format(code=code),
                    })
                if r.status_code in (401, 403):
                    # het session/token -> refresh roi thu lai
                    self.refresh(code)
                    payload["__RequestVerificationToken"] = self._token
                    last_err = f"HTTP {r.status_code}"
                    continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict):  # loi dang dict
                    raise RuntimeError(f"API tra ve loi: {str(data)[:200]}")
                return data
            except (requests.RequestException, RuntimeError,
                    json.JSONDecodeError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"fetch {code} [{from_d}..{to_d}] that bai sau "
            f"{MAX_RETRY} lan thu: {last_err}")


# ----------------------------------------------------------------------------
# PARSER
# ----------------------------------------------------------------------------
def parse_trading_date(val) -> str:
    """'/Date(1793898000000)/' hoac int ms -> 'YYYY-MM-DD' (gio VN)."""
    if isinstance(val, str):
        ms = int(re.sub(r"\D", "", val))
    else:
        ms = int(val)
    return datetime.fromtimestamp(ms / 1000, VN_TZ).strftime("%Y-%m-%d")


def chunk_dates(start: date, end: date):
    """Chia khoang ngay thanh cac chunk 1 nam (tu moi nhat ve cu)."""
    chunks = []
    cur_end = end
    while cur_end >= start:
        cur_start = max(date(cur_end.year, 1, 1), start)
        chunks.append((cur_start, cur_end))
        cur_end = date(cur_end.year - 1, 12, 31)
    return chunks


# ----------------------------------------------------------------------------
# FETCH 1 MA
# ----------------------------------------------------------------------------
def fetch_ticker(vs: VietstockSession, ticker: str, start: date, end: date,
                 quiet: bool = False):
    """Lay toan bo gia 1 ma trong khoang [start, end] -> DataFrame."""
    import pandas as pd

    rows = {}
    for c_start, c_end in chunk_dates(start, end):
        data = vs.price_history(ticker, c_start.isoformat(), c_end.isoformat())
        for r in data:
            d = parse_trading_date(r["TradingDate"])
            if d not in rows:
                rows[d] = {
                    "ticker": ticker,
                    "date": d,
                    "open": r.get("OpenPrice"),
                    "high": r.get("HighestPrice"),
                    "low": r.get("LowestPrice"),
                    "close": r.get("ClosePrice"),
                    "volume": r.get("TotalVol"),
                    "exchange": r.get("Exchange"),
                }
        if not quiet:
            print(f"    {ticker}: chunk {c_start}..{c_end} -> {len(data)} dong")
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(sorted(rows.values(), key=lambda x: x["date"]))
    # pct giong DB goc: close/prev_close - 1 (phan so)
    df["pct"] = df["close"].astype(float).pct_change(fill_method=None)
    # don vi gia giong DB goc: nghin dong (Vietstock tra VND)
    for c in ["open", "high", "low", "close"]:
        df[c] = (df[c].astype(float) / 1000).round(3)
    df["volume"] = df["volume"].astype("float64")
    return df


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Lay gia hang ngay 26 ma ngan hang tu Vietstock.")
    ap.add_argument("--tickers", default=",".join(BANKING_TICKERS),
                    help="Danh sach ma, phan cach boi dau phay.")
    ap.add_argument("--start", default="2022-01-01",
                    help="Ngay bat dau (mac dinh 2022-01-01).")
    ap.add_argument("--end", default=date.today().isoformat(),
                    help="Ngay ket thuc (mac dinh hom nay).")
    ap.add_argument("--outdir", default="data/raw/vietstock",
                    help="Thu muc chua CSV tung ma.")
    ap.add_argument("--parquet-out",
                    default="data/processed/vietstock_price_daily.parquet",
                    help="Duong dan parquet gop.")
    ap.add_argument("--csv-out",
                    default="data/processed/vietstock_price_daily.csv",
                    help="Duong dan CSV gop de xem Excel.")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Nghi giua 2 request (giay) - de lich su voi server.")
    ap.add_argument("--skip-csv", action="store_true",
                    help="Khong ghi CSV tung ma / CSV gop.")
    args = ap.parse_args()

    import pandas as pd

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.parquet_out).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("FETCH GIA NGAN HANG - VIETSTOCK (lich-su-gia)")
    print("=" * 64)
    print(f"Tickers : {len(tickers)} ma")
    print(f"Khoang  : {start} -> {end}")
    print(f"Sleep   : {args.sleep}s/request")

    vs = VietstockSession(sleep_sec=args.sleep)
    vs.refresh(tickers[0])
    print("Session : OK (cookie + token da nhan)\n")

    frames = []
    ok, fail = [], []
    for i, t in enumerate(tickers, 1):
        print(f"[{i:>2}/{len(tickers)}] {t}")
        try:
            df = fetch_ticker(vs, t, start, end)
            if df.empty:
                print("    ! Khong co du lieu - bo qua")
                fail.append(t)
                continue
            if not args.skip_csv:
                df.to_csv(outdir / f"{t}.csv", index=False,
                          encoding="utf-8-sig")
            frames.append(df)
            ok.append(t)
            print(f"    OK: {len(df)} ngay "
                  f"({df['date'].min()} -> {df['date'].max()})")
        except Exception as e:
            print(f"    LOI: {repr(e)[:180]}")
            fail.append(t)

    if not frames:
        print("\nKhong lay duoc du lieu nao. Kiem tra internet / endpoint.")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    all_df.to_parquet(args.parquet_out, index=False)
    if not args.skip_csv:
        all_df.to_csv(args.csv_out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 64)
    print(f"XONG: {len(ok)}/{len(tickers)} ma | "
          f"{len(all_df):,} dong | "
          f"{all_df['date'].min()} -> {all_df['date'].max()}")
    print(f"Parquet gop : {args.parquet_out}")
    if not args.skip_csv:
        print(f"CSV gop     : {args.csv_out}")
    if fail:
        print(f"MA THAT BAI : {', '.join(fail)}  (chay lai rieng cac ma nay)")
    print("Buoc tiep: python scripts/build_banking_daily.py")


if __name__ == "__main__":
    main()
