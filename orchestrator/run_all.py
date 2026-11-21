# -*- coding: utf-8 -*-
"""
orchestrator/run_all.py — TỔNG VẬN HÀNH FSE
============================================
Chạy MỘT file, tự động kích hoạt toàn bộ chuỗi của repo:

  [1] Crawl tin              : python -m scripts.run_collector
                               -> data/parsed/crawled_data_*.json
  [2] Inference best.pt      : python -m scripts.recompute_sentiment
                               (checkpoints/fse_sft/best.pt, PhoBERT)
                               -> data/processed/sentiment_daily.parquet
  [3] Giá VN-Quant (DB)      : python -m scripts.extract_banking_daily
                               (data/raw/vn_quant/master_quant_database.db)
                               -> data/processed/banking_daily.parquet
  [4] Giá Vietstock          : fetch_price_vietstock.py + build_banking_daily.py
                               (nếu đã có bảng giá thì bỏ qua; có cả nhánh
                                fallback đọc banking_daily_vietstock.csv)
                               -> data/processed/banking_daily_vietstock.parquet
  [5] Ghép S_t 2 nguồn       : nội bộ orchestrator
      (Model = best.pt, Gemini = Sentiment_NLP trong master DB,
       lấy thêm lịch sử từ s_t_daily_model_vs_gemini.csv ở gốc repo)
                               -> data/processed/s_t_daily_merged.csv
  [6] Join K4 + tương quan   : scripts/k4_refresh_vietstock.py
      Pearson/Spearman          -> banking_with_sentiment_vietstock.parquet
                               -> k4_vietstock_correlations.csv
  [7] Tương quan mở rộng     : python -m scripts.join_banking_sentiment
      (8 cặp + lag 0..5 +        python -m scripts.correlation_analysis
       scatter + rolling)        -> data/processed/correlation/*
  [8] HỒI QUY f(S_t) —       : scripts/fit_response_curve.py
      ANN + đạo hàm + ngưỡng    -> plot_<source>_<target>.png  (6 hình)
      toi han                   -> curve_*.csv, threshold_summary.csv
  [9] Thu hoạch              : gom mọi sản phẩm về orchestrator/output/
      run_<timestamp>/ + run_report.md (bảng tương quan + ngưỡng + list hình)
  [10] KIỂM CHỨNG LIÊN QUAN  : scripts/assess_relevance.py (mặc định CÓ, bỏ qua
      ngành ngân hàng            bằng --skip-relevance; KHÔNG đụng chuỗi chính)
      theo tier 1-4              -> data/processed/relevance_audit/
                                 (phân bố tier + S_t_all vs S_t_bank + hình
                                  hồi quy nguồn BankFilter để đối chiếu)

Cách chạy (từ GỐC repo, không cần kích hoạt venv):
    python orchestrator\\run_all.py
Tuỳ chọn:
    --skip-crawl          bỏ qua bước cào tin (dùng data/parsed sẵn có)
    --skip-inference      bỏ qua bước inference best.pt
    --skip-fetch-price    KHÔNG bao giờ gọi Vietstock (dùng dữ liệu giá sẵn có)
    --skip-relevance      bỏ qua bước 10 kiểm chứng liên quan ngành
    --sources cafef,vnexpress   giới hạn nguồn crawl
    --outdir <dir>        nơi thu hoạch (mặc định orchestrator/output)
    --dry-run             chỉ in lệnh sẽ chạy, không chạy thật

Yêu cầu: đã đặt checkpoints/fse_sft/best.pt và
data/raw/vn_quant/master_quant_database.db vào máy (2 file này không nằm
trên git). Thi 1 trong 2 thì pipeline vẫn chạy được phần còn lại và báo rõ.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np  # noqa: F401  (giữ tương thích môi trường repo)
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_DIR = Path(__file__).resolve().parent

BEST_PT = REPO_ROOT / "checkpoints" / "fse_sft" / "best.pt"
MASTER_DB = REPO_ROOT / "data" / "raw" / "vn_quant" / "master_quant_database.db"
ROOT_ST_CSV = REPO_ROOT / "s_t_daily_model_vs_gemini.csv"

BANK_DB_PQ = REPO_ROOT / "data" / "processed" / "banking_daily.parquet"
BANK_VS_PQ = REPO_ROOT / "data" / "processed" / "banking_daily_vietstock.parquet"
BANK_VS_CSV = REPO_ROOT / "data" / "processed" / "banking_daily_vietstock.csv"
SENT_DAILY_PQ = REPO_ROOT / "data" / "processed" / "sentiment_daily.parquet"
MERGED_ST_CSV = REPO_ROOT / "data" / "processed" / "s_t_daily_merged.csv"
K4_PQ = REPO_ROOT / "data" / "processed" / "banking_with_sentiment_vietstock.parquet"
JOIN_DB_PQ = REPO_ROOT / "data" / "processed" / "banking_with_sentiment.parquet"
FALLBACK_PQ = REPO_ROOT / "data" / "processed" / "banking_with_sentiment_fallback.parquet"

CURVE_DIR = REPO_ROOT / "data" / "processed" / "response_curves"
CORR_DIR = REPO_ROOT / "data" / "processed" / "correlation"
RELEVANCE_DIR = REPO_ROOT / "data" / "processed" / "relevance_audit"
RELEVANCE_BF_PQ = RELEVANCE_DIR / "banking_with_sentiment_bankfilter.parquet"

STEP_RESULTS: list[dict] = []  # name, status, note


# --------------------------------------------------------------------------- #
# 0) Tiện ích
# --------------------------------------------------------------------------- #
def reexec_into_venv() -> None:
    """Tự chuyển vào python của venv repo nếu có (không cần activate)."""
    if os.environ.get("FSE_ORCH_REEXEC") == "1":
        return
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    cand = REPO_ROOT / "venv" / sub / exe
    if cand.exists() and Path(sys.executable).resolve() != cand.resolve():
        print(f"[orchestrator] Chuyển sang python của venv: {cand}")
        rc = subprocess.call([str(cand), str(Path(__file__).resolve()), *sys.argv[1:]],
                             env=dict(os.environ, FSE_ORCH_REEXEC="1"))
        sys.exit(rc)


def py() -> list[str]:
    return [sys.executable]


def sub_env() -> dict:
    return dict(os.environ, PYTHONIOENCODING="utf-8",
                PYTHONUTF8="1", FSE_ORCH_REEXEC="1")


def mark(name: str, status: str, note: str = "") -> None:
    STEP_RESULTS.append({"step": name, "status": status, "note": note})
    icon = {"OK": "+", "SKIP": "-", "FAIL": "x", "FALLBACK": "~"}[status]
    print(f"\n[{icon}] Bước {name}: {status} {('- ' + note) if note else ''}")


def run_cmd(cmd: list, log: "LogFile", timeout: int = 3600) -> bool:
    """Chạy lệnh con, stream output ra màn hình + log. True nếu rc=0."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}")
    log.write(f"\n$ {printable}\n")
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(REPO_ROOT), env=sub_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
    except Exception as e:  # noqa: BLE001
        log.write(f"KHONG KHOI CHAY DUOC: {e}\n")
        print(f"  Khong khoi chay duoc: {e}")
        return False
    t0 = time.time()
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            print(f"  {line}", end="")
            log.write(line)
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        log.write("TIMEOUT\n")
        print("  TIMEOUT — đã dừng bước này.")
        return False
    dt = time.time() - t0
    log.write(f"rc={rc} ({dt:.1f}s)\n")
    return rc == 0


class LogFile:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write(self, text: str) -> None:
        self._fh.write(text)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# Các bước pipeline
# --------------------------------------------------------------------------- #
def step_crawl(args, log: LogFile) -> None:
    if args.skip_crawl:
        mark("crawl", "SKIP", "--skip-crawl")
        return
    cmd = [*py(), "-m", "scripts.run_collector"]
    if args.sources:
        cmd += ["--sources", args.sources]
    ok = run_cmd(cmd, log, timeout=1800)
    mark("crawl", "OK" if ok else "FAIL",
         "" if ok else "mạng/nguồn lỗi — dùng lại data/parsed sẵn có")


def step_inference(args, log: LogFile) -> None:
    if args.skip_inference:
        mark("inference", "SKIP", "--skip-inference")
        return
    if not BEST_PT.exists():
        mark("inference", "SKIP", f"thiếu {BEST_PT.relative_to(REPO_ROOT)}")
        return
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:  # noqa: BLE001
        mark("inference", "SKIP", f"thiếu thư viện: {e.__class__.__name__}")
        return
    ok = run_cmd([*py(), "-m", "scripts.recompute_sentiment"], log, timeout=7200)
    mark("inference", "OK" if ok else "FAIL")


def step_price_db(args, log: LogFile) -> None:
    if not MASTER_DB.exists():
        mark("price_db", "SKIP", f"thiếu {MASTER_DB.relative_to(REPO_ROOT)}")
        return
    ok = run_cmd([*py(), "-m", "scripts.extract_banking_daily"], log, timeout=1800)
    mark("price_db", "OK" if ok else "FAIL")


def step_price_vietstock(args, log: LogFile) -> str:
    """Đảm bảo BANK_VS_PQ tồn tại. Trả về cách đạt được."""
    if BANK_VS_PQ.exists() and not args.force_rebuild_price:
        mark("price_vietstock", "SKIP", "đã có banking_daily_vietstock.parquet")
        return "cached"
    if BANK_VS_CSV.exists():
        df = pd.read_csv(BANK_VS_CSV)
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        need = {"trade_date", "total_volume", "avg_return", "avg_range"}
        if need.issubset(df.columns):
            df.to_parquet(BANK_VS_PQ, index=False)
            mark("price_vietstock", "OK",
                 f"fallback CSV -> parquet ({len(df)} ngày, không cần mạng)")
            return "csv-fallback"
    if args.skip_fetch_price:
        mark("price_vietstock", "SKIP", "--skip-fetch-price và không có dữ liệu sẵn")
        return "missing"
    ok = run_cmd([*py(), "scripts/fetch_price_vietstock.py"], log, timeout=3600)
    if not ok:
        mark("price_vietstock", "FAIL", "fetch Vietstock lỗi (mạng?)")
        return "missing"
    ok = run_cmd([*py(), "scripts/build_banking_daily.py"], log, timeout=1800)
    mark("price_vietstock", "OK" if ok else "FAIL",
         "fetch + build" if ok else "build lỗi")
    return "fresh" if ok else "missing"


def step_merge_st(args) -> bool:
    """Ghép S_t hai nguồn: Model (best.pt) + Gemini (master DB).

    Ưu tiên dữ liệu tươi; lấy thêm lịch sử từ s_t_daily_model_vs_gemini.csv.
    """
    print("\n" + "=" * 68)
    print("[5] GHÉP S_t HAI NGUỒN (Model = best.pt, Gemini = vnquant DB)")
    print("=" * 68)
    model: dict = {}
    gemini: dict = {}
    src_model, src_gemini = [], []

    # (a) lịch sử từ CSV gốc repo (S_t = Gemini, S_model = Model theo K3/K4)
    if ROOT_ST_CSV.exists():
        try:
            df = pd.read_csv(ROOT_ST_CSV)
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            date_col = next((c for c in df.columns
                             if str(c).lower() in ("trade_date", "date", "ngay")),
                            df.columns[0])
            model_col = next((c for c in df.columns if "model" in str(c).lower()), None)
            gem_col = "S_t" if "S_t" in df.columns and date_col != "S_t" else None
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            if model_col:
                model.update(dict(zip(df[date_col],
                                      pd.to_numeric(df[model_col], errors="coerce"))))
                src_model.append(f"{ROOT_ST_CSV.name}:{model_col}")
            if gem_col:
                gemini.update(dict(zip(df[date_col],
                                       pd.to_numeric(df[gem_col], errors="coerce"))))
                src_gemini.append(f"{ROOT_ST_CSV.name}:{gem_col}")
            print(f"  CSV lịch sử: {len(df)} ngày ({df[date_col].min()} -> "
                  f"{df[date_col].max()})")
        except Exception as e:  # noqa: BLE001
            print(f"  Bỏ qua CSV lịch sử ({e.__class__.__name__}: {e})")

    # (b) Model mới từ inference best.pt (ghi đè ngày trùng)
    if SENT_DAILY_PQ.exists():
        sd = pd.read_parquet(SENT_DAILY_PQ)
        sd["date"] = pd.to_datetime(sd["date"]).dt.strftime("%Y-%m-%d")
        model.update(dict(zip(sd["date"],
                              pd.to_numeric(sd["S_t"], errors="coerce"))))
        src_model.append(f"sentiment_daily.parquet (best.pt, {len(sd)} ngày)")
        print(f"  Model (best.pt, mới): {len(sd)} ngày "
              f"({sd['date'].min()} -> {sd['date'].max()})")
    else:
        print("  [!] Chưa có sentiment_daily.parquet — Model chỉ có phần lịch sử CSV")

    # (c) Gemini từ master DB (ghi đè ngày trùng)
    if MASTER_DB.exists():
        conn = sqlite3.connect(str(MASTER_DB))
        g = pd.read_sql_query(
            "SELECT date, AVG(Sentiment_NLP) AS S_t FROM UNIFIED_RESEARCH_MATRIX "
            "WHERE Sentiment_NLP IS NOT NULL GROUP BY date ORDER BY date", conn)
        conn.close()
        g["date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
        gemini.update(dict(zip(g["date"],
                               pd.to_numeric(g["S_t"], errors="coerce"))))
        src_gemini.append(f"master_quant_database.db AVG(Sentiment_NLP) ({len(g)} ngày)")
        print(f"  Gemini (vnquant DB): {len(g)} ngày")
    else:
        print("  [!] Thiếu master DB — Gemini chỉ có phần lịch sử CSV")

    rows = sorted(set(model) | set(gemini))
    if not rows:
        mark("merge_st", "FAIL", "không có nguồn S_t nào")
        return False
    out = pd.DataFrame({
        "trade_date": rows,
        "S_t_model": [model.get(d) for d in rows],
        "S_t_gemini": [gemini.get(d) for d in rows],
    }).dropna(how="all", subset=["S_t_model", "S_t_gemini"])
    MERGED_ST_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MERGED_ST_CSV, index=False, encoding="utf-8")
    mark("merge_st", "OK",
         f"{len(out)} ngày | model: {'; '.join(src_model) or 'trống'} | "
         f"gemini: {'; '.join(src_gemini) or 'trống'}")
    return True


def step_k4(args, log: LogFile) -> bool:
    if not BANK_VS_PQ.exists():
        mark("k4_join", "SKIP", "không có bảng giá Vietstock")
        return False
    if not MERGED_ST_CSV.exists() and not step_merge_st(args):
        mark("k4_join", "SKIP", "không có S_t để join")
        return False
    cmd = [*py(), "scripts/k4_refresh_vietstock.py",
           "--st-csv", "data/processed/s_t_daily_merged.csv"]
    ok = run_cmd(cmd, log, timeout=1800)
    mark("k4_join", "OK" if ok else "FAIL")
    return ok


def step_correlation(args, log: LogFile) -> None:
    # (a) join đường DB (Gemini) nếu đủ dữ liệu
    if MASTER_DB.exists() and BANK_DB_PQ.exists():
        ok = run_cmd([*py(), "-m", "scripts.join_banking_sentiment"], log,
                     timeout=1800)
        if not ok:
            print("  join DB lỗi — vẫn thử correlation với parquet sẵn có")
    # (b) correlation mở rộng: 8 cặp Pearson/Spearman + lag + scatter + rolling
    if JOIN_DB_PQ.exists():
        ok = run_cmd([*py(), "-m", "scripts.correlation_analysis"], log,
                     timeout=1800)
        mark("correlation", "OK" if ok else "FAIL",
             "8 cặp + lag + scatter + rolling")
    else:
        mark("correlation", "SKIP", "thiếu banking_with_sentiment.parquet")


def step_curves(args, log: LogFile) -> bool:
    """Hồi quy f(S_t) — yêu cầu cột S_t_source trong input."""
    data_path = None
    if K4_PQ.exists():
        data_path = K4_PQ
    elif JOIN_DB_PQ.exists():
        # thêm cột S_t_source (S_t của đường DB = Gemini) cho script chạy được
        df = pd.read_parquet(JOIN_DB_PQ)
        df["S_t_source"] = "Gemini"
        df.to_parquet(FALLBACK_PQ, index=False)
        data_path = FALLBACK_PQ
    if data_path is None:
        mark("response_curves", "FAIL", "không có dataset sentiment+giá nào")
        return False
    rel = str(data_path.relative_to(REPO_ROOT)).replace("\\", "/")
    ok = run_cmd([*py(), "scripts/fit_response_curve.py", "--data", rel],
                 log, timeout=7200)
    mark("response_curves", "OK" if ok else "FAIL", f"input={data_path.name}")
    return ok


def step_relevance(args, log: LogFile) -> None:
    """[10] Kiểm chứng mức liên quan ngành — CHỈ ĐỌC, không đổi S_t chính.

    Thất bại cũng KHÔNG ảnh hưởng các bước khác (non-blocking).
    """
    if args.skip_relevance:
        mark("relevance_audit", "SKIP", "--skip-relevance")
        return
    ok = run_cmd([*py(), "-m", "scripts.assess_relevance"], log, timeout=3600)
    if not ok:
        mark("relevance_audit", "FAIL", "xem log — audit không chặn pipeline")
        return
    if RELEVANCE_BF_PQ.exists():
        ok2 = run_cmd([*py(), "scripts/fit_response_curve.py", "--data",
                       "data/processed/relevance_audit/"
                       "banking_with_sentiment_bankfilter.parquet",
                       "--outdir", "data/processed/relevance_audit/response_curves",
                       "--sources", "BankFilter"], log, timeout=7200)
        mark("relevance_audit", "OK" if ok2 else "FALLBACK",
             "tier + S_t_bank + hình BankFilter" if ok2
             else "audit OK, hình BankFilter lỗi (xem log)")
    else:
        mark("relevance_audit", "OK",
             "phân bố tier (không có điểm từng bài)")


def step_collect(args) -> Path | None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else ORCH_DIR / "output" / f"run_{ts}"
    if outdir.exists():
        outdir = ORCH_DIR / "output" / f"run_{ts}_{int(time.time()) % 1000}"
    outdir.mkdir(parents=True, exist_ok=True)

    def copy_into(src_dir: Path, pattern: str, subdir: str) -> list:
        dest = outdir / subdir
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in sorted(src_dir.glob(pattern)):
            shutil.copy2(f, dest / f.name)
            copied.append(dest / f.name)
        return copied

    plots_curves, curves_csv, thresholds = [], [], []
    if CURVE_DIR.exists():
        plots_curves = copy_into(CURVE_DIR, "plot_*.png", "response_curves")
        curves_csv = copy_into(CURVE_DIR, "curve_*.csv", "response_curves")
        thresholds = copy_into(CURVE_DIR, "threshold_summary.csv", "response_curves")
    scatter, rolling, corr_csv = [], [], []
    if CORR_DIR.exists():
        corr_csv = copy_into(CORR_DIR, "correlation_*.csv", "correlation")
        scatter = copy_into(CORR_DIR, "scatter_*.png", "correlation")
        rolling = copy_into(CORR_DIR, "rolling_correlation.png", "correlation")
    k4_csv = []
    if (REPO_ROOT / "data/processed/k4_vietstock_correlations.csv").exists():
        k4_csv = copy_into(REPO_ROOT / "data/processed",
                           "k4_vietstock_correlations.csv", "correlation")

    # ---- gom cả thư mục relevance_audit (nếu có) ----
    if RELEVANCE_DIR.exists():
        dest_ra = outdir / "relevance_audit"
        shutil.copytree(RELEVANCE_DIR, dest_ra, dirs_exist_ok=True)

    # -------- run_report.md --------
    lines = ["# FSE — Báo cáo chạy tổng vận hành",
             "",
             f"- Thời điểm chạy: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
             f"- Repo: `{REPO_ROOT}`",
             "",
             "## Trạng thái các bước",
             "", "| Bước | Trạng thái | Ghi chú |", "|---|---|---|"]
    for r in STEP_RESULTS:
        lines.append(f"| {r['step']} | {r['status']} | {r['note']} |")

    def md_table(csv_path: Path, max_rows: int = 40) -> str:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:  # noqa: BLE001
            return f"_(không đọc được {csv_path.name}: {e})_"
        df = df.head(max_rows)
        head = "| " + " | ".join(str(c) for c in df.columns) + " |"
        sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                         for row in df.itertuples(index=False))
        return f"_{csv_path.name}_\n\n{head}\n{sep}\n{body}"

    k4_file = outdir / "correlation" / "k4_vietstock_correlations.csv"
    corr_file = outdir / "correlation" / "correlation_results.csv"
    thr_file = outdir / "response_curves" / "threshold_summary.csv"
    lines += ["", "## Tương quan Pearson/Spearman (K4 — 2 nguồn S_t)", "",
              md_table(k4_file) if k4_file else "_(chưa có)_"]
    lines += ["", "## Tương quan mở rộng (8 cặp, đường DB)", "",
              md_table(corr_file) if corr_file else "_(chưa có)_"]
    lines += ["", "## Ngưỡng tối hạn f(S_t) (hồi quy ANN)", "",
              md_table(thr_file) if thr_file else "_(chưa có)_"]

    # ---- section relevance audit (sau khi def rel để dùng được rel()) ----
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(outdir)).replace("\\", "/")
        except ValueError:
            return str(p)

    rel_md = outdir / "relevance_audit" / "relevance_summary.md"
    if rel_md.exists():
        lines += ["", "---", "", "## Kiểm chứng mức liên quan ngành ngân hàng "
                  "(relevance audit)", "",
                  "_(S_t gốc của chuỗi chính KHÔNG bị thay đổi bởi phần này)_", "",
                  rel_md.read_text(encoding="utf-8", errors="replace")]
        rel_curves = sorted((outdir / "relevance_audit").rglob("plot_*.png"))
        if rel_curves:
            lines += ["", "### Hình hồi quy nguồn BankFilter (chỉ tin Tier 1+2)", ""]
            for p in rel_curves:
                lines.append(f"- ![BankFilter]({rel(p)})")

    lines += ["", "## Hình hồi quy f(S_t) — sản phẩm cuối", ""]
    for p in plots_curves:
        lines.append(f"- ![f(S_t)]({rel(p)})")
    lines += ["", "## Hình tương quan", ""]
    for p in scatter + rolling:
        lines.append(f"- ![]({rel(p)})")

    report = outdir / "run_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return outdir


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def preflight() -> list:
    print("=" * 68)
    print("PREFLIGHT — kiểm tra tài sản đầu vào")
    print("=" * 68)
    checks = [
        ("best.pt (trọng lượng model)", BEST_PT),
        ("master_quant_database.db (vnquant)", MASTER_DB),
        ("s_t_daily_model_vs_gemini.csv (lịch sử)", ROOT_ST_CSV),
        ("banking_daily_vietstock.parquet", BANK_VS_PQ),
        ("banking_daily_vietstock.csv (fallback)", BANK_VS_CSV),
        ("banking_with_sentiment.parquet (đường DB)", JOIN_DB_PQ),
    ]
    missing = []
    for label, path in checks:
        ok = path.exists()
        print(f"  [{'OK ' if ok else 'thiếu'}] {label}")
        if not ok:
            missing.append(label)
    for d in ("data/parsed", "data/processed"):
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FSE tổng vận hành — 1 file, ra hình hồi quy")
    ap.add_argument("--skip-crawl", action="store_true")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--skip-fetch-price", action="store_true",
                    help="không gọi Vietstock, chỉ dùng dữ liệu giá sẵn có")
    ap.add_argument("--force-rebuild-price", action="store_true")
    ap.add_argument("--skip-relevance", action="store_true",
                    help="bỏ qua bước 10 kiểm chứng mức liên quan ngành ngân hàng")
    ap.add_argument("--sources", default="", help="nguồn crawl, VD: cafef,vnexpress")
    ap.add_argument("--outdir", default="", help="thư mục thu hoạch sản phẩm")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reexec_into_venv()
    os.chdir(REPO_ROOT)

    print("=" * 68)
    print("FSE TỔNG VẬN HÀNH — crawl -> best.pt + vnquant -> hồi quy f(S_t)")
    print("=" * 68)
    preflight()

    log = LogFile(ORCH_DIR / "logs" /
                  f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    if args.dry_run:
        print("\n[DRY-RUN] Các lệnh sẽ được chạy:")
        for name, cmd in [
            ("crawl", [*py(), "-m", "scripts.run_collector"]
             + (["--sources", args.sources] if args.sources else [])),
            ("inference", [*py(), "-m", "scripts.recompute_sentiment"]),
            ("price_db", [*py(), "-m", "scripts.extract_banking_daily"]),
            ("price_vietstock", [*py(), "scripts/fetch_price_vietstock.py"]),
            ("build_daily", [*py(), "scripts/build_banking_daily.py"]),
            ("k4_join", [*py(), "scripts/k4_refresh_vietstock.py",
                         "--st-csv", "data/processed/s_t_daily_merged.csv"]),
            ("join_db", [*py(), "-m", "scripts.join_banking_sentiment"]),
            ("correlation", [*py(), "-m", "scripts.correlation_analysis"]),
            ("response_curves", [*py(), "scripts/fit_response_curve.py",
                                 "--data", "<auto>"]),
            ("relevance_audit", [*py(), "-m", "scripts.assess_relevance"]),
            ("relevance_curves", [*py(), "scripts/fit_response_curve.py",
                                  "--data",
                                  "data/processed/relevance_audit/"
                                  "banking_with_sentiment_bankfilter.parquet",
                                  "--sources", "BankFilter"]),
        ]:
            print(f"  [{name}] $ {' '.join(str(c) for c in cmd)}")
        print("\n[DRY-RUN] Kết thúc. Bỏ --dry-run để chạy thật.")
        return 0

    step_crawl(args, log)
    step_inference(args, log)
    step_price_db(args, log)
    step_price_vietstock(args, log)
    step_merge_st(args)
    step_k4(args, log)
    step_correlation(args, log)
    curves_ok = step_curves(args, log)
    step_relevance(args, log)
    outdir = step_collect(args)

    log.close()

    # ------- Tổng kết -------
    print("\n" + "=" * 68)
    print("TỔNG KẾT")
    print("=" * 68)
    for r in STEP_RESULTS:
        print(f"  [{r['status']:8s}] {r['step']:16s} {r['note']}")

    if outdir:
        print(f"\nSản phẩm đã gom về: {outdir}")
        print("  - run_report.md                 (bảng tương quan + ngưỡng + ảnh)")
        plots = sorted((outdir / "response_curves").glob("plot_*.png")) \
            if (outdir / "response_curves").exists() else []
        for p in plots:
            print(f"  - {p.relative_to(outdir)}")
        ra = outdir / "relevance_audit" / "relevance_summary.md"
        if ra.exists():
            print("  - relevance_audit/relevance_summary.md  (kiểm chứng liên quan "
                  "ngành — S_t gốc không đổi)")
    if not curves_ok:
        print("\n[!] Lưu ý: bước hồi quy chưa chạy được — kiểm tra phần SKIP/FAIL "
              "ở trên (thường do thiếu best.pt hoặc thiếu bảng giá).")

    fail_any = any(r["status"] == "FAIL" for r in STEP_RESULTS)
    return 1 if fail_any else 0


if __name__ == "__main__":
    sys.exit(main())
