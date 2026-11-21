# -*- coding: utf-8 -*-
"""
01_evaluate_model.py
====================
Đánh giá mô hình PhoBERT sentiment trên test set và sinh báo cáo đầy đủ:
- F1-macro, F1 per-class, precision/recall per-class
- Accuracy, balanced accuracy
- Confusion matrix (CSV + heatmap PNG)
- Classification report (CSV)
- Cohen's Kappa (so với random guess baseline)

Mục đích: lấp số F1-macro = 0.XX placeholder trong README và báo cáo, đồng thời
đưa ra confusion matrix để phân tích nhược điểm mô hình.

Input
-----
--test       : data/processed/test_3class.parquet (mặc định)
                Phải có cột ``text``, ``label`` (gold int 0/1/2),
                và ``label_pred`` (predicted int 0/1/2)
                HOẶC ``probs`` (numpy array shape [N, 3]) để argmax.
--labels     : file JSON ánh xạ int → tên lớp (mặc định {0: "neg", 1: "neu", 2: "pos"})

Output (data/processed/model_eval/)
- metrics.json
- classification_report.csv
- confusion_matrix.csv
- confusion_matrix.png

Cách chạy:
    python -m scripts.01_evaluate_model
    python -m scripts.01_evaluate_model --test data/processed/test_3class.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# UTF-8 stdout cho dấu tiếng Việt trên Windows pipe
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


DEFAULT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def _resolve_predictions(df: pd.DataFrame) -> np.ndarray:
    if "label_pred" in df.columns:
        return df["label_pred"].to_numpy()
    if "probs" in df.columns:
        probs = np.stack(df["probs"].to_numpy())
        return probs.argmax(axis=1)
    raise ValueError(
        "DataFrame phải có cột 'label_pred' hoặc 'probs'. "
        "Nếu chỉ có logits, đổi tên cột thành 'label_pred' hoặc chạy argmax trước."
    )


def evaluate(df: pd.DataFrame, label_map: dict[int, str]) -> dict:
    y_true = df["label"].to_numpy()
    y_pred = _resolve_predictions(df)
    labels_sorted = sorted(label_map.keys())
    target_names = [label_map[i] for i in labels_sorted]

    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels_sorted, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels_sorted, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels_sorted, zero_division=0)
    precision_per = precision_score(y_true, y_pred, average=None, labels=labels_sorted, zero_division=0)
    recall_per = recall_score(y_true, y_pred, average=None, labels=labels_sorted, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Random guess baseline (uniform): F1_macro = mean of per-class F1 under random
    # Per-class F1 under random ≈ 2*p / (1 + p) where p = class prevalence
    p_class = np.array([(y_true == i).mean() for i in labels_sorted])
    f1_random = np.where(p_class > 0, 2 * p_class / (1 + p_class), 0)
    f1_macro_random = float(np.mean(f1_random))

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    report = classification_report(
        y_true, y_pred, labels=labels_sorted, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    # Test thống kê: F1-macro có vượt random baseline không (approx via bootstrap)
    rng = np.random.default_rng(42)
    boot_f1 = np.empty(500)
    n = len(y_true)
    for b in range(500):
        idx = rng.integers(0, n, size=n)
        boot_f1[b] = f1_score(y_true[idx], y_pred[idx], average="macro",
                              labels=labels_sorted, zero_division=0)
    boot_lo, boot_hi = np.quantile(boot_f1, [0.025, 0.975])

    return {
        "n_test": int(n),
        "labels": label_map,
        "f1_macro": float(f1_macro),
        "f1_macro_ci_lo": float(boot_lo),
        "f1_macro_ci_hi": float(boot_hi),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": {label_map[i]: float(v) for i, v in zip(labels_sorted, f1_per)},
        "precision_per_class": {label_map[i]: float(v) for i, v in zip(labels_sorted, precision_per)},
        "recall_per_class": {label_map[i]: float(v) for i, v in zip(labels_sorted, recall_per)},
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "cohen_kappa": float(kappa),
        "f1_macro_random_baseline": f1_macro_random,
        "improvement_vs_random": float(f1_macro - f1_macro_random),
        "support": {label_map[i]: int((y_true == i).sum()) for i in labels_sorted},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": target_names,
        "classification_report": report,
    }


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — PhoBERT sentiment")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình PhoBERT sentiment trên test set")
    parser.add_argument("--test", default="data/processed/test_3class.parquet")
    parser.add_argument("--labels", default=None, help="JSON ánh xạ int→tên lớp")
    parser.add_argument("--out", default="data/processed/model_eval")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"[LỖI] Không tìm thấy test set: {test_path}", file=sys.stderr)
        return 2

    df = pd.read_parquet(test_path) if test_path.suffix == ".parquet" else pd.read_csv(test_path)
    if args.label_col not in df.columns:
        print(f"[LỖI] Thiếu cột '{args.label_col}' trong test set.", file=sys.stderr)
        return 2

    label_map = DEFAULT_LABELS
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as f:
            raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}

    print(f"[01] Đánh giá trên {len(df)} mẫu test, labels = {label_map}")
    metrics = evaluate(df, label_map)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # classification_report.csv
    rep_df = pd.DataFrame(metrics["classification_report"]).T
    rep_df.to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig")

    # confusion_matrix.csv + png
    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=metrics["confusion_matrix_labels"],
                        columns=metrics["confusion_matrix_labels"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion(cm, metrics["confusion_matrix_labels"], out_dir / "confusion_matrix.png")

    # In tóm tắt
    print("\n=== Tóm tắt đánh giá ===")
    print(f"F1-macro         : {metrics['f1_macro']:.4f}  (95% CI [{metrics['f1_macro_ci_lo']:.4f}, {metrics['f1_macro_ci_hi']:.4f}])")
    print(f"F1-weighted      : {metrics['f1_weighted']:.4f}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen's kappa    : {metrics['cohen_kappa']:.4f}")
    print(f"F1 random base   : {metrics['f1_macro_random_baseline']:.4f}")
    print(f"Improvement      : +{metrics['improvement_vs_random']:.4f}")
    print("\nF1 per-class:")
    for k, v in metrics["f1_per_class"].items():
        print(f"  {k:10s} : F1={v:.4f}  P={metrics['precision_per_class'][k]:.4f}  R={metrics['recall_per_class'][k]:.4f}  n={metrics['support'][k]}")

    print(f"\nĐã ghi: {out_dir}/metrics.json, classification_report.csv, confusion_matrix.csv/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
