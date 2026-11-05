"""
Đo chất lượng nhãn người gán so với model: Cohen's kappa + confusion + per-class.

Vì sao cần kappa thay vì % khớp: % khớp bị bóp méo bởi trùng hợp ngẫu nhiên
(66% câu neutral → model "đoán neutral" khớp 66% mà vô dụng). Kappa hiệu chỉnh
phần trùng hợp đó; > 0.6 là khá, > 0.8 là rất tốt.

Input: file xlsx do export_for_annotation.py xuất ra, người đã điền human_label.
Chạy từ gốc repo:
    python -m scripts.grade_annotation_agreement --file data/gold/review_batch_20261104.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ["negative", "neutral", "positive"]


def cohens_kappa(a: np.ndarray, b: np.ndarray, n_classes: int = 3) -> float:
    """Kappa giữa 2 dãy nhãn. Pure function — không phụ thuộc sklearn."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    mask = ~(pd.isna(a) | pd.isna(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).sum()) / n
    pe = sum(
        (np.sum(a == c) / n) * (np.sum(b == c) / n) for c in range(n_classes)
    )
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


def agreement_report(human, model) -> dict:
    human, model = np.asarray(human, int), np.asarray(model, int)
    kappa = cohens_kappa(human, model)
    raw_acc = float((human == model).sum()) / max(1, len(human))
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        sel = human == c
        per_class[name] = {
            "n": int(sel.sum()),
            "model_agrees_pct": round(float((model[sel] == c).mean() * 100), 1) if sel.sum() else None,
        }
    return {
        "n_annotated": int(len(human)),
        "raw_agreement": round(raw_acc, 4),
        "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
        "interpretation": (
            "rất tốt" if kappa >= 0.8 else
            "khá tốt" if kappa >= 0.6 else
            "trung bình — nên rà tiêu chí gán nhãn" if kappa >= 0.4 else
            "yếu — nhãn chưa đáng tin, xem lại tiêu chí/đào tạo annotator"
        ) if not np.isnan(kappa) else "không đủ dữ liệu",
        "per_class_human": per_class,
        "confusion_matrix": [[int(((human == i) & (model == j)).sum())
                              for j in range(3)] for i in range(3)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Đo agreement người vs model")
    parser.add_argument("--file", required=True, help="xlsx đã được người điền human_label")
    parser.add_argument("--human-col", default="human_label")
    parser.add_argument("--model-col", default=None,
                        help="Cột nhãn model (predicted_label). Nếu không có → dùng "
                             "model_confidence để ước lượng thô hoặc yêu cầu recompute.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[LỖI] Không tìm thấy file: {path}", file=sys.stderr)
        return 2
    df = pd.read_excel(path)
    if args.human_col not in df.columns:
        print(f"[LỖI] Thiếu cột '{args.human_col}' — người chưa gán nhãn? (có: {list(df.columns)})",
              file=sys.stderr)
        return 2
    human = pd.to_numeric(df[args.human_col], errors="coerce").dropna()
    df_valid = df.loc[human.index]

    if args.model_col and args.model_col in df.columns:
        model = pd.to_numeric(df_valid[args.model_col], errors="coerce")
        model = model.fillna(-1).astype(int)
    elif "predicted_label" in df.columns:
        model = pd.to_numeric(df_valid["predicted_label"], errors="coerce").fillna(-1).astype(int)
    else:
        print("[LỖI] Không tìm thấy cột nhãn model (predicted_label / --model-col).",
              file=sys.stderr)
        return 2

    keep = model >= 0
    report = agreement_report(human[keep].to_numpy(), model[keep].to_numpy())
    report["source"] = str(path)

    out = Path(args.out) if args.out else path.with_name(path.stem + "_agreement.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")

    print(f"Số câu đã gán nhãn : {report['n_annotated']}")
    print(f"Khớp thô           : {report['raw_agreement']:.1%}")
    print(f"Cohen's kappa      : {report['cohens_kappa']} ({report['interpretation']})")
    print("Confusion (hàng=người, cột=model):", report["confusion_matrix"])
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
