"""
Nạp checkpoint + đánh giá mô hình trên bất kỳ split nào.

Sửa so với bản cũ:
- Bản cũ KHÔNG HỀ chạm vào test set trong quá trình train/eval. Mọi lần
  retrain tự động giờ phải sinh báo cáo test đầy đủ (per-class + confusion
  matrix) — đây là "biên bản" để orchestrator so champion/challenger.
- load_checkpoint() tương thích cả checkpoint mới ({"state_dict", "meta"})
  lẫn rebuild model từ metadata (backbone / factory) — API serving có thể
  dùng luôn hàm này.

Dùng từ thư mục gốc repo:
  python -m src.training.evaluate \
      --checkpoint checkpoints/fse_sft/selected/best.pt \
      --config config/train_config.yaml --split test
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

logger = logging.getLogger("fse.evaluate")

CLASS_NAMES = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------------------
# Model building / loading
# ---------------------------------------------------------------------------
def build_model(model_cfg: dict) -> nn.Module:
    """Dựng model từ config.

    model_cfg["factory"] là import path (vd "src.models.fse_model:build_model")
    để cắm kiến trúc tùy chỉnh (vd AttentionPooling của repo). Nếu null →
    dùng AutoModelForSequenceClassification chuẩn của HuggingFace.
    """
    factory = model_cfg.get("factory")
    if factory:
        module_path, _, attr = factory.partition(":")
        if not module_path or not attr:
            raise ValueError(f'factory phải có dạng "module:function", nhận được: {factory!r}')
        mod = importlib.import_module(module_path)
        builder = getattr(mod, attr)
        logger.info("Dựng model từ factory: %s", factory)
        return builder(model_cfg)
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_cfg["backbone"], num_labels=int(model_cfg.get("num_labels", 3))
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Nạp checkpoint chuẩn hóa {"state_dict", "meta"} → (model, meta).

    Model được rebuild từ meta (ưu tiên factory, sau đó backbone) rồi mới
    load trọng số — tránh lệch kiến trúc giữa lúc train và lúc nạp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise ValueError(
            f"Checkpoint {path} không đúng định dạng mới ({{'state_dict', 'meta'}}). "
            "Checkpoint định dạng cũ (state_dict trần) không thể rebuild model — hãy train lại."
        )

    meta = ckpt.get("meta", {})
    model_cfg = {
        "backbone": meta.get("backbone", "VinAI/phobert-base-v2"),
        "num_labels": meta.get("num_labels", 3),
        "factory": meta.get("factory"),
    }
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, meta


# ---------------------------------------------------------------------------
# Dự đoán + báo cáo
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device):
    """Chạy suy luận trên loader → (y_true, y_pred, y_prob)."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        logits = out.logits if hasattr(out, "logits") else out
        all_probs.append(torch.softmax(logits.float(), dim=-1).cpu())
        all_labels.append(batch["labels"].cpu())
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred = probs.argmax(axis=1)
    return y_true, y_pred, probs


def build_report(y_true, y_pred, y_prob, class_names=CLASS_NAMES) -> dict:
    """Báo cáo đánh giá đầy đủ: accuracy, F1, per-class, confusion matrix."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {}
    for i, name in enumerate(class_names):
        row = rep.get(str(i), {})
        per_class[name] = {
            "precision": round(float(row.get("precision", 0.0)), 4),
            "recall": round(float(row.get("recall", 0.0)), 4),
            "f1": round(float(row.get("f1-score", 0.0)), 4),
            "support": int(row.get("support", 0)),
        }
    return {
        "n_samples": int(len(y_true)),
        "accuracy": round(float(rep["accuracy"]), 4),
        "f1_macro": round(float(rep["macro avg"]["f1-score"]), 4),
        "f1_weighted": round(float(rep["weighted avg"]["f1-score"]), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": list(class_names),
        # Độ tự tin trung bình của nhãn argmax (hữu ích khi S_t dùng nhãn cứng)
        "mean_confidence": round(float(y_prob.max(axis=1).mean()), 4),
    }


def report_to_markdown(report: dict, title: str = "Báo cáo đánh giá") -> str:
    """Chuyển report JSON thành bảng Markdown (để đọc nhanh/đính kèm PR)."""
    lines = [
        f"# {title}",
        "",
        f"- Số mẫu: **{report['n_samples']}**",
        f"- Accuracy: **{report['accuracy']:.4f}** | F1-macro: **{report['f1_macro']:.4f}** "
        f"| F1-weighted: **{report['f1_weighted']:.4f}**",
        f"- Độ tự tin trung bình (argmax): {report['mean_confidence']:.4f}",
        "",
        "## Chi tiết theo lớp",
        "",
        "| Lớp | Precision | Recall | F1 | Số mẫu |",
        "|---|---|---|---|---|",
    ]
    for name in report["class_names"]:
        pc = report["per_class"][name]
        lines.append(
            f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |"
        )
    lines += ["", "## Confusion matrix (hàng = thật, cột = dự đoán)", ""]
    header = "| đúng\\\\dự đoán | " + " | ".join(report["class_names"]) + " |"
    sep = "|---" * (len(report["class_names"]) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for i, name in enumerate(report["class_names"]):
        row = " | ".join(str(v) for v in report["confusion_matrix"][i])
        lines.append(f"| {name} | {row} |")
    lines.append("")
    return "\n".join(lines)


def save_report(report: dict, out_json: Path, out_md: Path | None = None,
                title: str = "Báo cáo đánh giá") -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if out_md is not None:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(report_to_markdown(report, title), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI: đánh giá một checkpoint trên bất kỳ split nào
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint FSE trên một split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--out-dir", default=None,
                        help="Mặc định: lưu cạnh checkpoint (eval_report_<split>.json/.md)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, seg_cfg, model_cfg = cfg["data"], cfg["segmentation"], cfg["model"]

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        args.device if args.device != "auto" else "cpu"
    )
    model, meta = load_checkpoint(args.checkpoint, device)

    from transformers import AutoTokenizer
    from ..data.dataset import FSEDataset3Class, Collator, load_parquet

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    file_map = {"train": data_cfg["train_file"], "val": data_cfg["val_file"], "test": data_cfg["test_file"]}
    df = load_parquet(
        Path(data_cfg["processed_dir"]) / file_map[args.split],
        data_cfg["text_col"], data_cfg["label_col"], data_cfg.get("min_text_chars", 0),
    )
    ds = FSEDataset3Class(
        df, tokenizer, max_len=model_cfg["max_len"],
        text_col=data_cfg["text_col"], label_col=data_cfg["label_col"],
        segmentation_mode=seg_cfg["mode"], segmentation_cache_dir=seg_cfg["cache_dir"],
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer))

    y_true, y_pred, y_prob = predict(model, loader, device)
    report = build_report(y_true, y_pred, y_prob)
    report["checkpoint"] = str(args.checkpoint)
    report["meta"] = {k: meta.get(k) for k in ("seed", "backbone", "saved_at", "data_hash")}

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    save_report(report, out_dir / f"eval_report_{args.split}.json",
                out_dir / f"eval_report_{args.split}.md",
                title=f"Đánh giá {args.split} — {Path(args.checkpoint).name}")
    logger.info("[%s] accuracy=%.4f | f1_macro=%.4f | n=%d",
                args.split, report["accuracy"], report["f1_macro"], report["n_samples"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
