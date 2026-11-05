"""
Drift monitor — theo dõi phân bố output model theo thời gian và cảnh báo.

Sinh ra để trả lời: "model train 1 lần trên dataset tĩnh, tin thực tế 2026 khác
phân phối 2016–2026 → khi nào phải retrain?" — câu trả lời là đo drift, không đoán.

Chỉ số: PSI (Population Stability Index) — công nghiệp, numpy thuần:
    PSI = Σ (p_i − q_i) · ln(p_i / q_i)
    < 0.1 ổn định | 0.1–0.25 theo dõi | > 0.25 lệch mạnh (rule of thumb)

Giám sát 3 tín hiệu mỗi tuần:
1. Tỷ lệ nhãn argmax (negative/neutral/positive) — drift phân bố lớp
2. Độ tự tin trung bình (mean max-prob) — model đang "lạc lối" dần
3. Số lượng tin/ngày — data pipeline có sống khỏe không (volume collapse)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PSI_STABLE, PSI_WATCH = 0.10, 0.25


# ---------------------------------------------------------------------------
# Các hàm thuần (testable, không phụ thuộc torch)
# ---------------------------------------------------------------------------
def psi_categorical(expected: list[float] | np.ndarray,
                    actual: list[float] | np.ndarray, eps: float = 1e-6) -> float:
    """PSI giữa 2 phân bố xác suất (tổng các bin = 1)."""
    e = np.clip(np.asarray(expected, dtype=float), eps, 1.0)
    a = np.clip(np.asarray(actual, dtype=float), eps, 1.0)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_continuous(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI cho biến liên tục (vd mean_confidence): chia bin theo baseline."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline, current = baseline[~np.isnan(baseline)], current[~np.isnan(current)]
    if len(baseline) < 30 or len(current) < 30:
        return float("nan")  # quá ít mẫu, không kết luận
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(baseline, bins=edges)[0] / max(1, len(baseline))
    a = np.histogram(current, bins=edges)[0] / max(1, len(current))
    return psi_categorical(e, a)


def weekly_stats(df: pd.DataFrame, date_col: str = "date",
                 label_col: str | None = "predicted_label",
                 conf_col: str | None = "confidence") -> pd.DataFrame:
    """Gom số liệu theo tuần ISO: số tin, tỷ lệ lớp, độ tự tin trung bình."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    week = d[date_col].dt.strftime("%G-W%V")  # tuần ISO
    rows = []
    for wk, g in d.groupby(week):
        row = {"week": wk, "n_articles": int(len(g))}
        if label_col and label_col in g.columns:
            props = g[label_col].value_counts(normalize=True)
            for cls in sorted(g[label_col].dropna().unique()):
                row[f"class_{int(cls)}_share"] = float(props.get(cls, 0.0))
        if conf_col and conf_col in g.columns:
            row["mean_confidence"] = float(g[conf_col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def check_drift(baseline_stats: pd.DataFrame, current_stats: pd.DataFrame) -> dict:
    """So tuần hiện tại với baseline — trả về PSI từng tín hiệu + cờ cảnh báo."""
    result: dict = {"signals": {}, "drifted": False}

    def _flag(name: str, value: float):
        level = ("stable" if value < PSI_STABLE
                 else "watch" if value < PSI_WATCH else "drift")
        if level == "drift":
            result["drifted"] = True
        result["signals"][name] = {"psi": round(value, 4), "level": level}

    # 1) phân bố lớp: baseline = trung bình toàn baseline window
    cls_cols = [c for c in baseline_stats.columns if c.startswith("class_")]
    if cls_cols and all(c in current_stats.columns for c in cls_cols):
        base = [float(baseline_stats[c].mean()) for c in cls_cols]
        cur = [float(current_stats[c].mean()) for c in cls_cols]
        _flag("class_distribution", psi_categorical(base, cur))

    # 2) độ tự tin
    if "mean_confidence" in baseline_stats and "mean_confidence" in current_stats:
        _flag("mean_confidence", psi_continuous(
            baseline_stats["mean_confidence"].to_numpy(),
            current_stats["mean_confidence"].to_numpy()))

    # 3) volume (so log-scale để ít nhạy với spike)
    if "n_articles" in baseline_stats and "n_articles" in current_stats:
        _flag("volume", psi_continuous(
            np.log1p(baseline_stats["n_articles"].to_numpy()),
            np.log1p(current_stats["n_articles"].to_numpy())))

    result["baseline_weeks"] = baseline_stats["week"].tolist()
    result["current_weeks"] = current_stats["week"].tolist()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame, baseline_weeks: int = 8, current_weeks: int = 2,
                 date_col: str = "date", label_col: str | None = "predicted_label",
                 conf_col: str | None = "confidence") -> dict:
    """Baseline = N tuần đầu của input; current = M tuần cuối."""
    stats = weekly_stats(df, date_col, label_col, conf_col)
    if len(stats) < baseline_weeks + current_weeks:
        baseline, current = stats.iloc[:-current_weeks], stats.iloc[-current_weeks:]
    else:
        baseline = stats.iloc[:baseline_weeks]
        current = stats.iloc[-current_weeks:]
    return check_drift(baseline, current)


def save_report(report: dict, out_json: Path, out_md: Path | None = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    if out_md:
        lines = ["# Drift report", ""]
        lines.append(f"- **Kết luận: {'⚠️ DRIFT' if report['drifted'] else '✅ Ổn định'}**")
        lines.append("")
        lines.append("| Tín hiệu | PSI | Mức |")
        lines.append("|---|---|---|")
        for name, s in report["signals"].items():
            lines.append(f"| {name} | {s['psi']} | {s['level']} |")
        out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Drift monitor cho output model FSE")
    parser.add_argument("--input", required=True,
                        help="Parquet có cột date + predicted_label + confidence "
                             "(vd output suy luận hằng ngày)")
    parser.add_argument("--baseline-weeks", type=int, default=8)
    parser.add_argument("--current-weeks", type=int, default=2)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--label-col", default="predicted_label")
    parser.add_argument("--conf-col", default="confidence")
    parser.add_argument("--out", default="reports/drift/drift_report.json")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Thoát mã 1 nếu drift → cron/alert bắt được")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("fse.drift")

    df = pd.read_parquet(args.input)
    report = build_report(df, args.baseline_weeks, args.current_weeks,
                          args.date_col, args.label_col or None, args.conf_col or None)
    out = Path(args.out)
    save_report(report, out, out.with_suffix(".md"))
    for name, s in report["signals"].items():
        logger.info("%s: PSI=%.4f (%s)", name, s["psi"], s["level"])
    logger.info("Kết luận: %s | báo cáo → %s",
                "DRIFT ⚠️" if report["drifted"] else "ổn định", out)
    return 1 if (args.fail_on_drift and report["drifted"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
