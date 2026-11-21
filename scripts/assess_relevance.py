# -*- coding: utf-8 -*-
"""assess_relevance.py — ĐÁNH GIÁ MỨC LIÊN QUAN NGÀNH NGÂN HÀNG (bước 10 orchestrator)

Mục đích
--------
KHÔNG đụng vào chuỗi chính đang chạy ổn. Script này CHỈ ĐỌC dữ liệu sẵn có,
gán nhãn mức liên quan (Relevance Tier 1-4 theo constants.py) cho từng bài
bằng quy tắc minh bạch (keyword + mã cổ phiếu ngân hàng), rồi trả lời câu hỏi
kiểm chứng:

    "S_t hiện tại tính trên TOÀN BỘ tin — nếu chỉ giữ tin ngân hàng (Tier 1+2)
     thì S_t thay đổi bao nhiêu? Kết luận có phụ thuộc cách lọc không?"

Tier theo spec (constants.py RELEVANCE_TIER_BASELINE_WEIGHTS):
    Tier 1 = 1.0  Nhắc trực tiếp 1 ngân hàng cụ thể (27 mã TARGET_BANK_TICKERS
                  hoặc tên thương hiệu: Vietcombank, BIDV, Techcombank...)
    Tier 2 = 0.7  Ngành tài chính - ngân hàng nói chung (ngân hàng, tín dụng,
                  lãi suất, nợ xấu, chứng khoán, trái phiếu, NHNN...)
    Tier 3 = 0.3  Kinh tế vĩ mô (lạm phát, GDP, xuất khẩu, tỷ giá, Fed...)
    Tier 4 = 0.0  Không liên quan / tin lẻ đường (không khớp từ điển nào)

Hai chế độ dữ liệu (tự phát hiện):
    DB mode     : master_quant_database.db / UNIFIED_RESEARCH_MATRIX có cột
                  Title (hoặc Content) + Sentiment_NLP từng bài
                  -> đủ để tính S_t_all vs S_t_bank theo ngày.
    Parsed mode : data/parsed/crawled_data_*.json (title + body + publish_time)
                  -> chỉ ra phân bố tier + tỷ lệ bài ngân hàng/ngày
                  (thêm --score-model nếu muốn chấm điểm bằng best.pt).

Đầu ra (data/processed/relevance_audit/):
    relevance_tier_distribution.csv   phân bố 4 tier
    tier_by_source.csv                phân bố tier theo nguồn tin
    bank_coverage.csv                 số bài nhắc từng ngân hàng (bằng chứng Tier 1)
    s_t_daily_bankfilter.csv          theo ngày: S_t_all, S_t_bank, S_t_bank_w,
                                      n_all, n_bank  (chỉ DB mode / có điểm)
    banking_with_sentiment_bankfilter.parquet
                                      schema tương thích fit_response_curve.py
                                      (S_t_source = "BankFilter")
    relevance_summary.md              báo cáo kiểm chứng + tương quan 2 đường

Chạy:
    python -m scripts.assess_relevance
    python -m scripts.assess_relevance --db <path.db> --outdir <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from constants import TARGET_BANK_TICKERS  # 27 mã chuẩn repo
except Exception:  # noqa: BLE001
    TARGET_BANK_TICKERS = [
        "ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "SHB", "SSB",
        "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "OCB", "NAB", "BVB", "VBB",
        "BAB", "NVB", "ABB", "KLB", "PGB", "SGB", "VAB",
    ]

TIER_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.3, 4: 0.0}

DEFAULT_DB = REPO_ROOT / "data" / "raw" / "vn_quant" / "master_quant_database.db"
PARSED_DIR = REPO_ROOT / "data" / "parsed"
PRICE_PQ = REPO_ROOT / "data" / "processed" / "banking_daily_vietstock.parquet"
PRICE_CSV = REPO_ROOT / "data" / "processed" / "banking_daily_vietstock.csv"
EXISTING_JOIN_PQ = REPO_ROOT / "data" / "processed" / "banking_with_sentiment.parquet"

# --------------------------------------------------------------------------- #
# Từ điển liên quan (so khớp trên văn bản BỎ DẤU + lowercase; mã CP giữ nguyên)
# --------------------------------------------------------------------------- #
# (Danh sách tên thương hiệu ngân hàng được build trong _build_dictionary bên dưới)

def _strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt + lowercase để so khớp từ điển (robust 2 kiểu gõ).

    Lưu ý: ký tự đ (U+0111) KHONG bị NFD phân rã — phải tự thay đ -> d.
    """
    nfkd = unicodedata.normalize("NFD", str(text))
    out = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return out.replace("đ", "d")


# Từ khóa NGÂN HÀNG ĐẶC ĐẶT (đủ tin cậy, khớp trên TOÀN VĂN title+body)
T2_STRONG_TERMS = [
    "ngan hang", "ngân hàng", "tctd", "to chuc tin dung", "tổ chức tín dụng",
    "ngan hang thuong mai", "ngân hàng thương mại", "tin dung", "tín dụng",
    "lai suat", "lãi suất", "no xau", "nợ xấu", "nhnn", "ngan hang nha nuoc",
    "ngân hàng nhà nước", "omo", "tin phieu", "tín phiếu", "trich lap",
    "trích lập", "bao lanh", "bảo lãnh", "chiet khau", "chiết khấu",
    "tai cap von", "tái cấp vốn", "room tin dung", "basel", "he so car",
    "du no", "dư nợ", "tra no", "trả nợ", "vay von", "vay vốn", "gui tiet kiem",
    "gửi tiết kiệm", "tiet kiem", "tiết kiệm", "the tin dung", "thẻ tín dụng",
    "atm", "swift", "lim tin dung", "han muc tin dung", "hạn mức tín dụng",
]

# Từ khóa TÀI CHÍNH RỘNG (de bị boilerplate công ty lái sang — chỉ khớp TIÊU ĐỀ)
T2_BROAD_TERMS = [
    "tai chinh", "tài chính", "chung khoan", "chứng khoán", "co phieu",
    "cổ phiếu", "trai phieu", "trái phiếu", "niem yet", "niêm yết", "ipo",
    "hose", "hnx", "upcom", "vn-index", "vnindex", "vn 30", "khoi ngoai",
    "khối ngoại", "thanh khoan", "thanh khoản", "chot phien", "chốt phiên",
    "phien giao dich", "phiên giao dịch", "san giao dich", "sàn giao dịch",
    "bao hiem", "bảo hiểm", "quy dau tu", "quỹ đầu tư", "quy mo", "quỹ mở",
    "so giao dich chung khoan", "sở giao dịch chứng khoán", "sbv", "doanh nghiep niem yet",
]

# Vĩ mô ĐẶC ĐẶT (khớp toàn văn)
T3_STRONG_TERMS = [
    "lam phat", "lạm phát", "cpi", "gdp", "grdp", "fdi", "ty gia", "tỷ giá",
    "xuat khau", "xuất khẩu", "nhap khau", "nhập khẩu", "can can thuong mai",
    "cán cân thương mại", "no cong", "nợ công", "no chinh phu", "nợ chính phủ",
    "pmi", "chinh sach tien te", "chính sách tiền tệ", "ngan sach nha nuoc",
    "ngân sách nhà nước", "tang truong kinh te", "tăng trưởng kinh tế",
    "kinh te vi mo", "kinh tế vĩ mô", "kinh te the gioi", "kinh tế thế giới",
]

# Vĩ mô RỘNG (de boilerplate — chỉ khớp TIÊU ĐỀ)
T3_BROAD_TERMS = [
    "tang truong", "tăng trưởng", "tien te", "tiền tệ", "thue", "thuế",
    "ngan sach", "ngân sách", "fed", "ecb", "usd", "nhip kinh te",
    "nhịp kinh tế", "nen kinh te", "nền kinh tế", "kinh te", "kinh tế",
    "thi truong vang", "thị trường vàng", "gia vang", "giá vàng", "dau tho",
    "dầu thô", "hang hoa", "hàng hóa",
]


def _build_dictionary():
    """Trả về 5 regex: mã CP, tên thương hiệu, T2 đậm đặc, T2 rộng, T3 đậm đặc, T3 rộng."""
    # Tier 1a: mã cổ phiếu — word-boundary, PHÂN BIỆT hoa/thường trên văn bản gốc
    ticker_re = re.compile(
        r"(?<![A-Za-z])(" + "|".join(sorted(TARGET_BANK_TICKERS, key=len, reverse=True))
        + r")(?![A-Za-z])"
    )
    # Tier 1b: tên thương hiệu ngân hàng (không dấu, lowercase)
    proper = [
        "vietcombank", "bidv", "vietinbank", "techcombank", "vpbank", "mb bank",
        "ngan hang quan doi", "agribank", "sacombank", "seabank", "eximbank",
        "hdbank", "tpbank", "maritimebank", "lienvietpostbank", "pvcombank",
        "bacabank", "abbank", "saigonbank", "pgbank", "vietabank",
        "kienlongbank", "vietbank", "nam a bank", "ngan hang nha nuoc",
        "ngan hang nha nuoc viet nam",
    ]
    proper = sorted(set(proper), key=len, reverse=True)
    proper_re = re.compile(
        r"(?<![a-z0-9])(" + "|".join(re.escape(p) for p in proper) + r")(?![a-z0-9])"
    )

    def list_re(terms):
        terms = sorted({_strip_accents(t) for t in terms}, key=len, reverse=True)
        return re.compile(
            r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in terms) + r")(?![a-z0-9])"
        )

    return (ticker_re, proper_re, list_re(T2_STRONG_TERMS), list_re(T2_BROAD_TERMS),
            list_re(T3_STRONG_TERMS), list_re(T3_BROAD_TERMS))


(TICKER_RE, PROPER_RE, T2_STRONG_RE, T2_BROAD_RE,
 T3_STRONG_RE, T3_BROAD_RE) = _build_dictionary()


def classify_article(title: str, body: str) -> tuple[int, list]:
    """Gán tier cho 1 bài. Trả về (tier, danh_sách_ngân_hàng_nhắc_đến).

    Thiết kế chống boilerplate: từ khóa RỘNG (tài chính/chứng khoán chung,
    tăng trưởng, thuế...) chỉ khớp trên TIÊU ĐỀ; từ khóa NGÂN HÀNG ĐẶC ĐẶT
    và mã/tên ngân hàng khớp trên toàn văn.
    """
    text = f"{title or ''}\n{body or ''}"
    banks = sorted(set(m.group(0) for m in TICKER_RE.finditer(text)))
    banks += sorted(set(m.group(0) for m in PROPER_RE.finditer(_strip_accents(text))))
    if banks:
        return 1, banks
    title_norm = _strip_accents(title or "")
    body_norm = _strip_accents(body or "")
    if T2_STRONG_RE.search(body_norm) or T2_BROAD_RE.search(title_norm):
        return 2, []
    if T3_STRONG_RE.search(body_norm) or T3_BROAD_RE.search(title_norm):
        return 3, []
    return 4, []


# --------------------------------------------------------------------------- #
# Nạp bài viết: DB mode hoặc parsed mode
# --------------------------------------------------------------------------- #
def load_articles_db(db_path: Path):
    """Đọc từng bài từ UNIFIED_RESEARCH_MATRIX (rows có Sentiment_NLP).

    Trả về DataFrame [date, title, body, source, score] hoặc None nếu thiếu cột.
    """
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(UNIFIED_RESEARCH_MATRIX)")]
    except Exception:  # noqa: BLE001
        conn.close()
        return None
    title_col = next((c for c in ("Title", "title", "headline", "article_title")
                      if c in cols), None)
    body_col = next((c for c in ("Content", "content", "body", "text",
                                 "article_content") if c in cols), None)
    score_col = next((c for c in ("Sentiment_NLP", "sentiment_nlp") if c in cols), None)
    src_col = next((c for c in ("Source", "source", "website") if c in cols), None)
    if title_col is None and body_col is None:
        print(f"  [!] {db_path.name}/UNIFIED_RESEARCH_MATRIX không có cột tiêu đề/nội dung")
        conn.close()
        return None
    sel = ["date"]
    sel += [title_col] if title_col else ["'' AS title"]
    sel += [body_col] if body_col else ["'' AS body"]
    sel += [score_col] if score_col else ["NULL AS score"]
    sel += [src_col] if src_col else ["'' AS source"]
    where = ("WHERE Sentiment_NLP IS NOT NULL" if score_col else
             f"WHERE {title_col} IS NOT NULL AND {title_col} != ''" if title_col
             else "")
    df = pd.read_sql_query(
        f"SELECT {', '.join(sel)} FROM UNIFIED_RESEARCH_MATRIX "
        + where, conn)
    conn.close()
    df.columns = ["date", "title", "body", "score", "source"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    print(f"  DB mode: {len(df)} bài có {score_col or '(chưa có điểm)'} "
          f"từ {df['date'].min()} -> {df['date'].max()}")
    return df


def load_articles_parsed(parsed_dir: Path) -> pd.DataFrame | None:
    files = sorted(parsed_dir.glob("crawled_data_*.json")) if parsed_dir.exists() else []
    if not files:
        return None
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for a in (data if isinstance(data, list) else []):
            date = str(a.get("publish_time") or a.get("discovered_time") or "")[:10]
            rows.append({
                "date": date,
                "title": a.get("title") or "",
                "body": (a.get("body") or a.get("content") or "")[:4000],
                "score": None,
                "source": a.get("source") or "",
            })
    df = pd.DataFrame(rows)
    df = df[df["date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    print(f"  Parsed mode: {len(df)} bài từ {len(files)} file crawl")
    return df if len(df) else None


# --------------------------------------------------------------------------- #
# Tổng hợp
# --------------------------------------------------------------------------- #
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Theo ngày: S_t_all (toàn bộ), S_t_bank (Tier<=2), S_t_bank_w (trọng số tier)."""
    recs = defaultdict(lambda: {"a": [], "b": [], "num_w": 0.0, "den_w": 0.0})
    for r in df.itertuples(index=False):
        d = getattr(r, "date")
        s = getattr(r, "score")
        tier = getattr(r, "tier")
        if pd.isna(s):
            continue
        recs[d]["a"].append(float(s))
        if tier in (1, 2):
            recs[d]["b"].append(float(s))
        w = TIER_WEIGHTS.get(tier, 0.0)
        if w > 0:
            recs[d]["num_w"] += float(s) * w
            recs[d]["den_w"] += w
    rows = []
    for d in sorted(recs):
        v = recs[d]
        rows.append({
            "trade_date": d,
            "S_t_all": round(float(np.mean(v["a"])), 4) if v["a"] else np.nan,
            "S_t_bank": round(float(np.mean(v["b"])), 4) if v["b"] else np.nan,
            "S_t_bank_w": round(v["num_w"] / v["den_w"], 4) if v["den_w"] else np.nan,
            "n_all": len(v["a"]),
            "n_bank": len(v["b"]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Đánh giá mức liên quan ngành ngân hàng")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--parsed-dir", default=str(PARSED_DIR))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "data" / "processed"
                                            / "relevance_audit"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("[10] ĐÁNH GIÁ LIÊN QUAN NGÀNH NGÂN HÀNG (relevance audit)")
    print("=" * 68)

    df = load_articles_db(Path(args.db))
    mode = "DB"
    if df is None or len(df) == 0:
        df = load_articles_parsed(Path(args.parsed_dir))
        mode = "PARSED"
    if df is None or len(df) == 0:
        print("  [!] Không có nguồn bài viết nào (DB không có cột văn bản,"
              " data/parsed rỗng).")
        (outdir / "relevance_summary.md").write_text(
            "# Relevance audit — KHÔNG CHẠY ĐƯỢC\n\n"
            "Không tìm thấy bài viết: master DB không có cột Title/Content "
            "và data/parsed/*.json không tồn tại (đã dọn dẹp?).\n",
            encoding="utf-8")
        return 0

    print(f"  Chế độ: {mode} — {len(df)} bài")
    tiers, bank_hits = [], []
    for r in df.itertuples(index=False):
        tier, banks = classify_article(r.title, r.body)
        tiers.append(tier)
        for b in banks:
            bank_hits.append(b)
    df["tier"] = tiers

    dist = (df["tier"].value_counts().reindex([1, 2, 3, 4]).fillna(0).astype(int)
            .rename_axis("tier").reset_index(name="n_articles"))
    dist["share_pct"] = (dist["n_articles"] / len(df) * 100).round(2)
    dist["trong_so"] = [TIER_WEIGHTS[t] for t in dist["tier"]]
    dist.to_csv(outdir / "relevance_tier_distribution.csv", index=False,
                encoding="utf-8-sig")

    if "source" in df.columns and df["source"].ne("").any():
        by_src = (df.groupby(["source", "tier"]).size().unstack(fill_value=0)
                  .reindex(columns=[1, 2, 3, 4], fill_value=0))
        by_src.to_csv(outdir / "tier_by_source.csv", encoding="utf-8-sig")

    bc = Counter(bank_hits).most_common()
    pd.DataFrame(bc, columns=["bank", "n_articles"]).to_csv(
        outdir / "bank_coverage.csv", index=False, encoding="utf-8-sig")

    has_scores = df["score"].notna().any()
    daily = aggregate_daily(df) if has_scores else pd.DataFrame(
        columns=["trade_date", "S_t_all", "S_t_bank", "S_t_bank_w", "n_all", "n_bank"])
    if not has_scores:
        comp = df.groupby("date")["tier"].agg(
            n_all="count", n_bank=lambda s: int((s <= 2).sum())).reset_index()
        comp.columns = ["trade_date", "n_all", "n_bank"]
        comp["bank_share_pct"] = (comp["n_bank"] / comp["n_all"] * 100).round(1)
        daily = comp[["trade_date", "n_all", "n_bank", "bank_share_pct"]]
    daily.to_csv(outdir / "s_t_daily_bankfilter.csv", index=False,
                 encoding="utf-8-sig")

    # ---- Parquet tương thích fit_response_curve (chỉ khi có điểm từng bài) ----
    parquet_made = False
    if has_scores:
        price = None
        if PRICE_PQ.exists():
            price = pd.read_parquet(PRICE_PQ)
        elif PRICE_CSV.exists():
            price = pd.read_csv(PRICE_CSV)
            price.columns = [str(c).strip().lstrip("\ufeff") for c in price.columns]
        if price is not None:
            price["trade_date"] = pd.to_datetime(
                price.iloc[:, 0]).dt.strftime("%Y-%m-%d")
            m = daily.merge(price, on="trade_date", how="inner")
            if len(m) >= 60:
                m["S_t"] = m["S_t_bank"]
                m["S_t_source"] = "BankFilter"
                m.to_parquet(outdir / "banking_with_sentiment_bankfilter.parquet",
                             index=False)
                parquet_made = True
                print(f"  Parquet BankFilter: {len(m)} ngày khớp bảng giá")

    # ---- Tương quan 2 đường + kiểm tra khớp S_t gốc ----
    corr_line = "_(chưa có điểm từng bài — chỉ có phân bố tier)_"
    if has_scores and len(daily) > 10:
        d = daily.dropna(subset=["S_t_all", "S_t_bank"])
        if len(d) > 10:
            r_p = d["S_t_all"].corr(d["S_t_bank"])
            r_s = d["S_t_all"].corr(d["S_t_bank"], method="spearman")
            corr_line = (f"Pearson r = **{r_p:.3f}**, Spearman ρ = **{r_s:.3f}** "
                         f"trên {len(d)} ngày")
    sanity_line = ""
    if mode == "DB" and EXISTING_JOIN_PQ.exists() and has_scores and len(daily) > 0:
        try:
            ex = pd.read_parquet(EXISTING_JOIN_PQ)[["trade_date", "S_t"]]
            chk = daily.merge(ex, on="trade_date", how="inner")
            if len(chk) > 10:
                diff = (chk["S_t_all"] - chk["S_t"]).abs().max()
                r = chk["S_t_all"].corr(chk["S_t"])
                if diff < 1e-3 and r > 0.9999:
                    verdict = ("audit đang đọc ĐÚNG dữ liệu của chuỗi chính "
                               "(chênh lệch chỉ ở mức làm tròn số thập phân).")
                else:
                    verdict = ("CẢNH BÁO: S_t_all KHÔNG khớp S_t gốc của "
                               "banking_with_sentiment.parquet — kiểm tra lại "
                               "bản DB / khoảng thời gian trước khi dùng kết luận.")
                sanity_line = (f"\nKiểm tra khớp chuỗi gốc: chênh lệch S_t_all vs "
                               f"S_t trong banking_with_sentiment.parquet tối đa "
                               f"= **{diff:.2e}**, Pearson r = **{r:.4f}** "
                               f"({len(chk)} ngày chung) — {verdict}")
        except Exception as e:  # noqa: BLE001
            sanity_line = f"\n(Không kiểm tra khớp được: {e})"

    t1 = int((df["tier"] == 1).sum())
    t2 = int((df["tier"] == 2).sum())
    share12 = (t1 + t2) / len(df) * 100
    top_banks = ", ".join(f"{b} ({n})" for b, n in bc[:8]) or "—"

    md = [
        "# Relevance audit — mức liên quan ngành ngân hàng của corpus",
        "",
        f"- Chế độ dữ liệu: **{mode}** | số bài: **{len(df):,}**",
        f"- Tier 1 (nhắc ngân hàng cụ thể): **{t1:,}** bài "
        f"({t1 / len(df) * 100:.1f}%)",
        f"- Tier 2 (ngành tài chính - ngân hàng): **{t2:,}** bài "
        f"({t2 / len(df) * 100:.1f}%)",
        f"- **Tổng Tier 1+2 = {share12:.1f}%** corpus",
        f"- Ngân hàng được nhắc nhiều nhất: {top_banks}",
        "",
        "## Phân bố tier",
        "",
        "| tier | n_articles | share_pct | trong_so |",
        "|---|---|---|---|"] + [
        f"| {int(r.tier)} | {int(r.n_articles):,} | {r.share_pct:.2f} | "
        f"{TIER_WEIGHTS[int(r.tier)]} |" for r in dist.itertuples(index=False)
    ] + [
        "",
        "## S_t toàn bộ vs S_t chỉ tin ngân hàng (Tier 1+2)",
        "",
        corr_line,
        sanity_line,
        "",
        "> Diễn giải: nếu tương quan hai đường cao (|r| > 0.8), S_t hiện tại "
        "(trung bình toàn bộ tin) và S_t lọc ngân hàng cho cùng hình thái — "
        "các kết luận kiểm định KHÔNG phụ thuộc việc chưa lọc liên quan. "
        "S_t gốc của chuỗi chính KHÔNG bị thay đổi bởi audit này.",
    ]
    (outdir / "relevance_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n  Phân bố tier: T1={t1} T2={t2} "
          f"T3={int((df['tier'] == 3).sum())} T4={int((df['tier'] == 4).sum())}")
    print(f"  Tier 1+2 = {share12:.1f}%")
    print(f"  Đầu ra -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
