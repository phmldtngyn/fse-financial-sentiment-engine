# -*- coding: utf-8 -*-
"""
k4_refresh_vietstock.py
=======================
Nap gia MOI tu Vietstock (banking_daily_vietstock.parquet) vao bai toan
tuong quan K4, khong can sua file nao khac:

  1. Tai bang gia nganh moi: data/processed/banking_daily_vietstock.parquet
     -> them log_volume = ln(total_volume), |S_t|
  2. S_t phien ban GEMINI: tu tinh AVG(Sentiment_NLP) theo ngay tu
     master DB (giong het join_banking_sentiment.py cu)
  3. S_t phien ban MODEL: tu tim file s_t_daily_model_vs_gemini.csv
     (da co tu lan chay K3/K4 truoc - khong can suy lai model).
     Khong tim thay thi chi chay phien ban Gemini va bao cach lay CSV.
  4. Tinh lai 3 cap tuong quan cho moi phien ban:
        S_t   <-> avg_return   (gia)
        |S_t| <-> log_volume   (khoi luong)
        S_t   <-> avg_range    (bien do trong ngay)
     Pearson + Spearman kem p-value.

Dau ra:
  data/processed/banking_with_sentiment_vietstock.parquet
  data/processed/k4_vietstock_correlations.csv

Cach chay:
  python scripts/k4_refresh_vietstock.py
  python scripts/k4_refresh_vietstock.py --st-csv "D:/FSE/s_t_daily_model_vs_gemini.csv"
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# So tham chieu K4 lan truoc (tren bang gia CU) - de doi chieu, khong phai ket qua moi
OLD_K4 = {
    "Gemini": {"S_t - avg_return": 0.161, "|S_t| - log_volume": 0.167,
               "S_t - avg_range": -0.140},
    "Model":  {"S_t - avg_return": 0.093, "|S_t| - log_volume": 0.163,
               "S_t - avg_range": -0.112},
}


def pearson_p(r: float, n: int) -> float:
    """p-value gan dung cho Pearson khi khong co scipy (xap xi normal, n lon)."""
    if n < 3 or abs(r) >= 1:
        return 0.0 if abs(r) >= 1 else float("nan")
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return math.erfc(abs(t) / math.sqrt(2))


def corr_pair(df: pd.DataFrame, col_a: str, col_b: str):
    """Pearson + Spearman + p cho 1 cap cot (bo NaN, la khong tham so)."""
    sub = df[[col_a, col_b]].dropna()
    n = len(sub)
    a, b = sub[col_a].to_numpy(), sub[col_b].to_numpy()
    if n < 3 or a.std() == 0 or b.std() == 0:
        return n, float("nan"), float("nan"), float("nan"), float("nan")
    pr = float(np.corrcoef(a, b)[0, 1])
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sr = float(np.corrcoef(ra, rb)[0, 1])
    try:
        from scipy import stats as st
        pp = float(st.pearsonr(a, b)[1])
        sp = float(st.spearmanr(a, b)[1])
    except Exception:
        pp = pearson_p(pr, n)
        sp = pearson_p(sr, n)
    return n, pr, pp, sr, sp


def find_st_csv(explicit: str):
    """Tim file S_t daily model vs gemini (mac dinh o goc project / data/)."""
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands += [Path("s_t_daily_model_vs_gemini.csv"),
              Path("data/processed/s_t_daily_model_vs_gemini.csv"),
              Path("data/raw/s_t_daily_model_vs_gemini.csv"),
              Path("data/s_t_daily_model_vs_gemini.csv")]
    for c in cands:
        if c.exists():
            return c
    return None


def detect_columns(df: pd.DataFrame):
    """Tu nhan cot ngay va cot S_t trong CSV."""
    date_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "date" in lc or "ngay" in lc or lc == "trade_date":
            date_col = c
            break
    model_col, gemini_col = None, None
    for c in df.columns:
        lc = str(c).lower()
        if "model" in lc and ("s_t" in lc or "st" in lc or "s_" in lc):
            model_col = c
        if "gemini" in lc and ("s_t" in lc or "st" in lc or "s_" in lc):
            gemini_col = c
    # truong hop ten cot la S_t_model / S_t_gemini da bat o tren;
    # neu khong, lay cot so thu 2 / 3 lam du phuong an
    if model_col is None and len(df.columns) >= 2:
        model_col = df.columns[1] if str(df.columns[1]).lower() != str(date_col).lower() else None
    return date_col, model_col, gemini_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banking",
                    default="data/processed/banking_daily_vietstock.parquet")
    ap.add_argument("--db",
                    default="data/raw/vn_quant/master_quant_database.db")
    ap.add_argument("--st-csv", default="",
                    help="Duong dan s_t_daily_model_vs_gemini.csv (tu chuyen).")
    ap.add_argument("--out",
                    default="data/processed/"
                            "banking_with_sentiment_vietstock.parquet")
    ap.add_argument("--corr-out",
                    default="data/processed/k4_vietstock_correlations.csv")
    args = ap.parse_args()

    print("=" * 68)
    print("K4 REFRESH - gia Vietstock moi + S_t (Gemini & Model)")
    print("=" * 68)

    # ---- 1) Bang gia nganh moi ----
    bank = pd.read_parquet(args.banking)
    bank["trade_date"] = pd.to_datetime(bank["trade_date"]).dt.strftime("%Y-%m-%d")
    bank["log_volume"] = np.log(bank["total_volume"].clip(lower=1))
    print(f"[1] Gia nganh moi : {len(bank):,} ngay "
          f"({bank['trade_date'].min()} -> {bank['trade_date'].max()})")

    # ---- 2+3) S_t gemini + model ----
    st_cols = {}  # name -> Series(date, S_t)
    csv_path = find_st_csv(args.st_csv)
    if csv_path:
        st = pd.read_csv(csv_path)
        st.columns = [str(c).strip() for c in st.columns]
        date_col, model_col, gemini_col = detect_columns(st)
        print(f"[2] Tim thay S_t CSV: {csv_path}")
        print(f"    Cot nhan ra     : ngay='{date_col}' "
              f"model='{model_col}' gemini='{gemini_col}'")
        st[date_col] = pd.to_datetime(st[date_col]).dt.strftime("%Y-%m-%d")
        if model_col is not None:
            st_cols["Model"] = st[[date_col, model_col]].rename(
                columns={date_col: "trade_date", model_col: "S_t"})
        if gemini_col is not None:
            st_cols["Gemini"] = st[[date_col, gemini_col]].rename(
                columns={date_col: "trade_date", gemini_col: "S_t"})
    if "Gemini" not in st_cols:
        if not Path(args.db).exists():
            print(f"[!] Khong thay {args.db} - bo qua S_t Gemini")
        else:
            import sqlite3
            conn = sqlite3.connect(args.db)
            g = pd.read_sql_query(
                "SELECT date, AVG(Sentiment_NLP) AS S_t FROM "
                "UNIFIED_RESEARCH_MATRIX WHERE Sentiment_NLP IS NOT NULL "
                "GROUP BY date ORDER BY date", conn)
            conn.close()
            g["trade_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
            st_cols["Gemini"] = g[["trade_date", "S_t"]]
            print(f"[2] S_t Gemini tu DB: {len(g):,} ngay")
    if not st_cols:
        raise SystemExit(
            "Khong co S_t nao. Tai file s_t_daily_model_vs_gemini.csv tu "
            "Drive (MyDrive/FSE_Project/) ve goc project, roi chay:\n"
            "  python scripts/k4_refresh_vietstock.py "
            "--st-csv duong_dan_file.csv")

    # ---- 4) Join + 3 cap tuong quan cho moi phien ban ----
    pairs = [("S_t - avg_return", "S_t", "avg_return"),
             ("|S_t| - log_volume", "abs_S_t", "log_volume"),
             ("S_t - avg_range", "S_t", "avg_range")]
    rows, merged_parts = [], {}
    for name, sdf in st_cols.items():
        df = bank.merge(sdf, on="trade_date", how="inner")
        df["abs_S_t"] = df["S_t"].abs()
        merged_parts[name] = df
        print(f"[3] Join {name:7s}: {len(df):,} ngay co ca gia + sentiment")
        for label, ca, cb in pairs:
            n, pr, pp, sr, sp = corr_pair(df, ca, cb)
            rows.append({"version": name, "pair": label, "n_days": n,
                         "pearson_r": round(pr, 4),
                         "p_pearson": f"{pp:.2e}" if pp == pp else "nan",
                         "spearman_rho": round(sr, 4),
                         "p_spearman": f"{sp:.2e}" if sp == sp else "nan"})

    corr = pd.DataFrame(rows)
    print("\n" + "=" * 68)
    print("KET QUA K4 MOI (gia Vietstock) - so voi lan truoc (gia cu)")
    print("=" * 68)
    for _, r in corr.iterrows():
        old = OLD_K4.get(r["version"], {}).get(r["pair"])
        old_txt = f" | cu: {old:+.3f}" if old is not None else ""
        print(f"  [{r['version']:7s}] {r['pair']:20s} "
              f"r={r['pearson_r']:+.3f} (p={r['p_pearson']}, "
              f"n={r['n_days']:,}, rho={r['spearman_rho']:+.3f}){old_txt}")

    # ---- 5) Luu ----
    out_frames = []
    for name, df in merged_parts.items():
        d = df.copy()
        d["S_t_source"] = name
        out_frames.append(d)
    out = pd.concat(out_frames, ignore_index=True)
    out.to_parquet(args.out, index=False)
    corr.to_csv(args.corr_out, index=False, encoding="utf-8-sig")
    print(f"\nOK -> {args.out}")
    print(f"OK -> {args.corr_out}")
    print("\nGhi chu: K3 (S_t model vs gemini, r=+0.729) KHONG phu thuoc gia "
          "nen khong can tinh lai. Cac con so 'cu' chi la tham chieu tu bang "
          "gia 4 nam cu de doi chieu.")


if __name__ == "__main__":
    main()
