"""
Kiểm tra luồng CLI end-to-end của bộ ops — tạo dữ liệu giả trong tmp, chạy
các script qua subprocess như người dùng thật, xác nhận output đúng.

Chạy từ gốc repo:  python -m pytest tests/test_cli_flows.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str], expect_rc: int = 0):
    proc = subprocess.run([PY, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == expect_rc, (
        f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}")
    return proc


@pytest.fixture(scope="module")
def news_df():
    rng = np.random.default_rng(42)
    subjects = ["Vietcombank", "Techcombank", "BIDV", "VPBank"]
    templates = ["{s} báo lợi nhuận tăng mạnh quý này", "{s} giảm sát giá sàn",
                 "{s} họp đại hội đồng cổ đông thường niên", "Giá vàng thế giới biến động"]
    rows = []
    base = pd.Timestamp("2026-03-02 09:30")   # thứ Hai
    for i in range(120):
        subject = subjects[i % len(subjects)]
        text = rng.choice(templates).format(s=subject)
        ts = base + pd.Timedelta(days=int(i // 20), hours=int(i % 20))
        if i > 0 and i % 17 == 0:              # chèn bản trùng để dedup bắt được
            text = rows[-1]["title"]
            ts = pd.Timestamp(rows[-1]["published_at"]) + pd.Timedelta(minutes=5)
        rows.append({"title": text, "text": f"{text}. Chi tiết về {subject} và thị trường tín dụng.",
                     "source": f"src{i % 3}", "url": f"https://example.com/{i}",
                     "published_at": ts.isoformat()})
    return pd.DataFrame(rows)


def test_full_flow(tmp_path, news_df):
    # ---- 1) apply_news_alignment → aligned parquet ----
    raw = tmp_path / "articles.parquet"
    news_df.to_parquet(raw, index=False)
    out_aligned = tmp_path / "articles_aligned.parquet"
    run(["scripts.apply_news_alignment", "--input", str(raw),
         "--holidays", "config/holidays_vn.yaml", "--ticker-map", "config/ticker_map.yaml",
         "--out", str(out_aligned)])
    aligned = pd.read_parquet(out_aligned)
    assert {"id", "tickers", "trading_date", "is_duplicate"} <= set(aligned.columns)
    assert (aligned["trading_date"].str.isnumeric() == False).all()  # dạng YYYY-MM-DD
    # tin 09:30 thứ Hai → phiên cùng ngày; tin 4h sáng (i%20 tạo giờ 0..19) → phiên theo quy tắc
    assert aligned["is_duplicate"].sum() >= 1
    matched = aligned[aligned["tickers"] != "[]"]
    assert len(matched) > 50          # 3/4 template có tên công ty
    assert "VCB" in json.dumps(aligned["tickers"].tolist())

    # ---- 2) upsert vào SQLite (idempotent: chạy 2 lần) ----
    db = tmp_path / "fse.db"
    run(["scripts.apply_news_alignment", "--input", str(raw),
         "--holidays", "config/holidays_vn.yaml", "--out", str(out_aligned),
         "--to-sqlite", str(db)])
    run(["scripts.apply_news_alignment", "--input", str(raw),
         "--holidays", "config/holidays_vn.yaml", "--out", str(out_aligned),
         "--to-sqlite", str(db)])
    from src.storage.sqlite_db import connect

    with connect(db) as conn:
        n1 = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    assert n1 == len(aligned)          # không nhân đôi

    # ---- 3) prepare_splits end-to-end ----
    src = tmp_path / "with_scores.parquet"
    n = 300
    dates = pd.date_range("2024-01-01", "2026-01-01", freq="D")[:n]
    pd.DataFrame({
        "text": [f"câu tin số {i} về ngân hàng" for i in range(n)],
        "score": np.random.default_rng(1).uniform(-1, 1, n),
        "date": dates,
    }).to_parquet(src, index=False)
    run(["scripts.prepare_splits", "--source", str(src), "--out-dir", str(tmp_path / "splits")])
    tr = pd.read_parquet(tmp_path / "splits" / "train_3class.parquet")
    te = pd.read_parquet(tmp_path / "splits" / "test_3class.parquet")
    assert set(tr.columns) == {"text", "label", "score", "date"}
    assert max(pd.to_datetime(tr["date"])) < min(pd.to_datetime(te["date"]))
    assert set(tr["label"].unique()) <= {0, 1, 2}

    # ---- 4) export_for_annotation (probs-col) + grade agreement ----
    conf = np.random.default_rng(2).uniform(0.3, 1.0, len(aligned))
    aligned["confidence"] = conf
    news_conf = tmp_path / "with_conf.parquet"
    aligned.to_parquet(news_conf, index=False)
    xlsx = tmp_path / "review_batch.xlsx"
    run(["scripts.export_for_annotation", "--news", str(news_conf),
         "--probs-col", "confidence", "--n", "40", "--out", str(xlsx)])
    annotated = pd.read_excel(xlsx, sheet_name="to_annotate")
    assert len(annotated) == 40
    annotated["human_label"] = np.random.default_rng(3).choice([0, 1, 2], 40)
    annotated["predicted_label"] = np.random.default_rng(4).choice([0, 1, 2], 40)
    xlsx2 = tmp_path / "review_filled.xlsx"
    annotated.to_excel(xlsx2, index=False)
    run(["scripts.grade_annotation_agreement", "--file", str(xlsx2)])
    assert (tmp_path / "review_filled_agreement.json").exists()

    # ---- 5) correlation_robust ----
    corr = pd.DataFrame({
        "ticker": ["VCB"] * 10, "feature": [f"f{i}" for i in range(10)],
        "corr": np.random.default_rng(5).uniform(-0.4, 0.4, 10),
        "p_value": np.random.default_rng(6).uniform(0.0, 0.5, 10),
    })
    corr_path = tmp_path / "corr.parquet"
    corr.to_parquet(corr_path, index=False)
    run(["scripts.correlation_robust", "--input", str(corr_path)])
    fdr = pd.read_parquet(tmp_path / "corr_fdr.parquet")
    assert "q_value_bh" in fdr.columns and (fdr["q_value_bh"] >= 0).all()
    summary = json.loads((tmp_path / "corr_fdr.json").read_text(encoding="utf-8"))
    assert summary["n_tests"] == 10

    # ---- 6) drift CLI (tạo predictions 20 tuần) ----
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-05-01", periods=140, freq="D")
    drift_input = tmp_path / "predictions.parquet"
    pd.DataFrame({
        "date": dates,
        "predicted_label": rng.choice([0, 1, 2], 140, p=[0.2, 0.6, 0.2]),
        "confidence": rng.uniform(0.5, 0.99, 140),
    }).to_parquet(drift_input, index=False)
    out_rep = tmp_path / "drift.json"
    run(["src.training.drift_monitor", "--input", str(drift_input),
         "--baseline-weeks", "12", "--current-weeks", "4", "--out", str(out_rep)])
    report = json.loads(out_rep.read_text(encoding="utf-8"))
    assert set(report["signals"]) == {"class_distribution", "mean_confidence", "volume"}

    # ---- 7) scheduler --run-once (job alignment) ----
    run(["scripts.scheduled_jobs", "--run-once", "news_alignment"], expect_rc=2)
    # expect 2 vì job command trỏ đường dẫn mặc định không tồn tại trong ROOT
    # (mục đích: xác minh cơ chế load config + subprocess hoạt động và báo rc đúng)
