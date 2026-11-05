"""
Active learning: xuất các tin mà model ÍT TỰ TIN nhất để người gán nhãn.

Cơ chế: chỉ ~20% gold set hiện tại từng được người rà soát; bộ này biến việc
rà soát thành vòng lặp tự động — mỗi tháng xuất batch 500 câu "model lúng túng
nhất" (đáng giá gán nhất), ưu tiên gán nhãn đúng chỗ cần thiết.

Chiến lược chọn mẫu:
- uncertain: max-prob thấp nhất (model phân vân giữa các lớp)
- entropy: phân bố xác suất đều nhất
- mixed (mặc định): 60% uncertain + 40% entropy + 10% random để không bỏ lớp hiếm

Output: data/gold/review_batch_<ngày>.xlsx — khớp flow for_review.xlsx có sẵn
trong repo (merge_annotations.py / finalize_gold.py). Cột human_label để người điền.

Chạy từ gốc repo:
    # Tự suy luận bằng checkpoint production:
    python -m scripts.export_for_annotation --checkpoint checkpoints/production/best.pt \
        --news data/processed/articles_aligned.parquet --n 500
    # Hoặc đã có cột confidence sẵn trong parquet:
    python -m scripts.export_for_annotation --news ... --probs-col confidence
"""

from __future__ import annotations

import argparse
import logging
import sys

# --- Windows an toàn: khi stdout/stderr là pipe/redirect, Python dùng codepage
# locale (vd cp1252) và sẽ crash khi in dấu tiếng Việt -> ép UTF-8 bắt buộc.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def select_for_annotation(probs: np.ndarray, n: int,
                          strategy: str = "mixed", seed: int = 42) -> list[int]:
    """Chọn index câu đáng gán nhãn nhất. Pure function — test được không cần model."""
    probs = np.asarray(probs, dtype=float)
    n_total = len(probs)
    n = min(n, n_total)
    rng = np.random.default_rng(seed)

    if strategy == "uncertain":
        return list(np.argsort(probs.max(axis=1))[:n])
    if strategy == "entropy":
        return list(np.argsort(_entropy(probs))[::-1][:n])

    # mixed: 50% uncertain + 40% entropy + 10% random (từ phần chưa chọn)
    n_unc, n_ent = int(n * 0.5), int(n * 0.4)
    chosen = list(dict.fromkeys(
        np.argsort(probs.max(axis=1))[:n_unc].tolist()
        + np.argsort(_entropy(probs))[::-1][:n_ent].tolist()))
    rest = list(set(range(n_total)) - set(chosen))
    n_random = min(max(0, n - len(chosen)), len(rest))
    chosen += rng.choice(rest, size=n_random, replace=False).tolist()
    return chosen[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Xuất batch gán nhãn (active learning)")
    parser.add_argument("--news", required=True, help="Parquet tin cần gán (chưa có nhãn)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint FSE (nếu không có cột confidence trong parquet)")
    parser.add_argument("--probs-col", default=None,
                        help="Cột chứa confidence/max-prob có sẵn (bỏ qua checkpoint)")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--strategy", choices=["mixed", "uncertain", "entropy"], default="mixed")
    parser.add_argument("--out", default=None,
                        help="Mặc định data/gold/review_batch_YYYYMMDD.xlsx")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("fse.active_learn")

    path = Path(args.news)
    if not path.exists():
        print(f"[LỖI] Không tìm thấy tin: {path}", file=sys.stderr)
        return 2
    df = pd.read_parquet(path)
    if args.text_col not in df.columns:
        print(f"[LỖI] Thiếu cột '{args.text_col}' (có: {list(df.columns)})", file=sys.stderr)
        return 2

    # ---- lấy xác suất dự đoán ----
    if args.probs_col and args.probs_col in df.columns:
        conf = pd.to_numeric(df[args.probs_col], errors="coerce").to_numpy()
        probs = np.stack([1 - conf, conf], axis=1)  # proxy 2 lớp, chỉ để xếp hạng
        logger.info("Dùng cột confidence có sẵn '%s' (không nạp model).", args.probs_col)
    elif args.checkpoint:
        try:
            import torch
            from torch.utils.data import DataLoader
            from transformers import AutoTokenizer

            from src.training.evaluate import build_report, load_checkpoint, predict
        except ImportError as exc:
            print(f"[LỖI] Cần torch/transformers để suy luận: {exc}", file=sys.stderr)
            return 2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_checkpoint(args.checkpoint, device)
        tokenizer = AutoTokenizer.from_pretrained(meta.get("backbone", "VinAI/phobert-base-v2"))
        try:
            from src.data.dataset import FSEDataset3Class, Collator

            ds = FSEDataset3Class(df, tokenizer, max_len=meta.get("max_len", 256),
                                  segmentation_mode="auto")
        except Exception:
            # dataset fse-training không có trong repo fallback: tokenize tay không pad
            from src.training.evaluate import predict as _p  # noqa: F401

            raise
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=Collator(tokenizer))
        _y, _pred, probs = predict(model, loader, device)
        logger.info("Đã suy luận %d câu bằng %s (device=%s)", len(df), args.checkpoint, device)
    else:
        print("[LỖI] Cần --checkpoint hoặc --probs-col để biết model tự tin ở đâu.", file=sys.stderr)
        return 2

    idx = select_for_annotation(probs, args.n, strategy=args.strategy)
    out_df = df.iloc[idx].copy()
    conf = probs[idx].max(axis=1)
    out_df["model_confidence"] = np.round(conf, 4)
    out_df["human_label"] = ""          # người điền: 0=negative, 1=neutral, 2=positive
    out_df["human_score"] = ""          # người điền: score [-1, 1] nếu muốn
    out_df["annotation_note"] = ""

    out = Path(args.out) if args.out else (
        Path("data/gold") / f"review_batch_{datetime.now(timezone.utc):%Y%m%d}.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="to_annotate")
        pd.DataFrame({
            "Hướng dẫn": [
                "human_label: 0 = negative, 1 = neutral, 2 = positive",
                "human_score: -1.0 (rất tiêu cực) → 1.0 (rất tích cực), 0 = trung tính",
                "Ưu tiên các câu model_confidence THẤP — chúng quyết định model tiến bộ",
                "Xong: python -m scripts.grade_annotation_agreement --file <file này>",
            ]
        }).to_excel(writer, index=False, sheet_name="instructions")
    logger.info("Xuất %d câu → %s", len(out_df), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
