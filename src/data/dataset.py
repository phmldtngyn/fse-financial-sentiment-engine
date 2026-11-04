"""
Dataset 3 lớp cho FSE + kiểm tra chất lượng split dữ liệu.

Sửa so với bản cũ (src/data/dataset.py):
1. TỰ ĐỘNG PHÁT HIỆN text chưa tách từ — PhoBERT yêu cầu input dạng
   "ngan_hang" (từ kép nối bằng "_"). Bản cũ tokenize thẳng text thô,
   là lỗi âm thầm làm giảm F1 nếu dữ liệu nguồn chưa pre-segmented.
   → mode "auto" sẽ tách từ bằng underthesea, có cache ra đĩa để các lần
   chạy sau (retrain định kỳ) không tốn thời gian.
2. Tokenize KHÔNG pad sẵn. Collator sẽ pad theo độ dài lớn nhất TRONG BATCH
   thay vì padding="max_length" cứng 256 ký tự → train nhanh hơn đáng kể
   với dữ liệu 50–800 ký tự.
3. verify_splits() — kiểm tra split theo thời gian + trùng lặp chéo các split
   (leakage), việc mà bản cũ hoàn toàn không có.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("fse.dataset")

CLASS_NAMES = ["negative", "neutral", "positive"]
# Tỷ lệ token chứa "_" để coi như text đã tách từ (từ kép trong tiếng Việt
# pre-segmented luôn nối bằng "_"; text thô rất hiếm khi có "_").
SEGMENTED_TOKEN_RATIO = 0.15


# ---------------------------------------------------------------------------
# Word segmentation (PhoBERT cần text đã tách từ)
# ---------------------------------------------------------------------------
def looks_segmented(texts: list[str], sample: int = 500) -> bool:
    """Phát hiện nhanh: text đã tách từ chưa? (dựa trên tỷ lệ token có '_')"""
    sample_texts = [t for t in texts if isinstance(t, str) and t.strip()][:sample]
    if not sample_texts:
        return False
    ratios = []
    for t in sample_texts:
        tokens = t.split()
        if not tokens:
            continue
        ratios.append(sum(1 for tok in tokens if "_" in tok) / len(tokens))
    if not ratios:
        return False
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    return median_ratio >= SEGMENTED_TOKEN_RATIO


class SegmentationCache:
    """Cache kết quả tách từ theo md5(text) — retrain sau chỉ xử lý câu mới."""

    def __init__(self, cache_dir: str | Path):
        self.path = Path(cache_dir) / "segmentation_cache.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._map: dict[str, str] = {}
        if self.path.exists():
            try:
                self._map = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # cache hỏng thì bỏ, tách lại từ đầu
                self._map = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Trả về (kết quả đã cache, danh sách index chưa có trong cache)."""
        out: list[str | None] = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            hit = self._map.get(self._key(t))
            if hit is None:
                missing.append(i)
            else:
                out[i] = hit
        return out, missing  # type: ignore[return-value]

    def put_many(self, texts: list[str], segmented: list[str]) -> None:
        for t, s in zip(texts, segmented):
            self._map[self._key(t)] = s
        self.path.write_text(
            json.dumps(self._map, ensure_ascii=False), encoding="utf-8"
        )


def segment_texts(texts: list[str], cache_dir: str | Path) -> list[str]:
    """Tách từ bằng underthesea, có cache ra đĩa (idempotent)."""
    try:
        from underthesea import word_tokenize
    except ImportError as exc:  # thông báo rõ ràng thay vì stack trace khó đọc
        raise ImportError(
            "Cần tách từ cho PhoBERT nhưng chưa cài underthesea. "
            "Chạy: pip install underthesea  (xem requirements-training.txt)"
        ) from exc

    cache = SegmentationCache(cache_dir)
    cached, missing = cache.get_many(texts)
    if missing:
        logger.info("Tách từ underthesea cho %d câu mới (cache: %d câu)...",
                    len(missing), len(texts) - len(missing))
        for i in missing:
            cached[i] = word_tokenize(texts[i], format="text")
        cache.put_many([texts[i] for i in missing], [cached[i] for i in missing])  # type: ignore[index]
    else:
        logger.info("Tất cả %d câu đã có trong cache segmentation.", len(texts))
    return cached  # type: ignore[return-value]


def ensure_segmented(
    texts: list[str], mode: str = "auto", cache_dir: str | Path = "data/cache/segmentation"
) -> list[str]:
    """Theo mode cấu hình: off → giữ nguyên; on → luôn tách; auto → detect rồi quyết định."""
    if mode == "off":
        return list(texts)
    if mode == "auto" and looks_segmented(texts):
        logger.info("Phát hiện dữ liệu ĐÃ tách từ sẵn → bỏ qua segmentation.")
        return list(texts)
    return segment_texts(texts, cache_dir)


# ---------------------------------------------------------------------------
# Đọc dữ liệu + kiểm tra split
# ---------------------------------------------------------------------------
def load_parquet(path: str | Path, text_col: str = "text", label_col: str = "label",
                 min_text_chars: int = 0) -> pd.DataFrame:
    """Đọc parquet, kiểm tra cột bắt buộc, lọc câu quá ngắn, reset index."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    df = pd.read_parquet(path)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise KeyError(f"File {path} thiếu cột bắt buộc '{col}'. "
                           f"Các cột hiện có: {list(df.columns)}")
    before = len(df)
    if min_text_chars > 0:
        df = df[df[text_col].astype(str).str.len() >= min_text_chars].reset_index(drop=True)
    if before - len(df) > 0:
        logger.info("%s: lọc bỏ %d câu < %d ký tự (còn %d câu).",
                    path.name, before - len(df), min_text_chars, len(df))
    return df.reset_index(drop=True)


def _norm_text(t: str) -> str:
    """Chuẩn hóa tối thiểu để so trùng lặp (không phân biệt hoa/thường/dấu câu)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()


def verify_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    date_col: str | None = None,
) -> list[str]:
    """Kiểm tra chất lượng split, trả về danh sách cảnh báo (list rỗng = ổn).

    Kiểm tra:
    1. Phân bố lớp ở từng split (cảnh báo nếu lớp nào < 30 mẫu ở test).
    2. Split theo thời gian: nếu có cột ngày, kiểm tra max(train) <= min(test)
       — nếu vi phạm nghĩa là split KHÔNG theo thời gian → nguy cơ leakage.
    3. Trùng lặp chéo split: câu giống hệt nhau (đã chuẩn hóa) xuất hiện ở cả
       train và test → đánh giá bị lạc quan.
    """
    warnings: list[str] = []
    name_df = {"train": train_df, "val": val_df, "test": test_df}

    # 1) Phân bố lớp
    for name, df in name_df.items():
        dist = df[label_col].value_counts().sort_index().to_dict()
        logger.info("Phân bố lớp [%s]: %s", name, dist)
    test_min = test_df[label_col].value_counts().min()
    if test_min < 30:
        warnings.append(
            f"Test set có lớp chỉ {test_min} mẫu (<30) → F1 per-class của lớp đó sẽ không tin cậy."
        )

    # 2) Split theo thời gian
    if date_col:
        have = all(date_col in df.columns for df in name_df.values())
        if have:
            t_max = pd.to_datetime(train_df[date_col]).max()
            v_min, v_max = (pd.to_datetime(val_df[date_col]).min(), pd.to_datetime(val_df[date_col]).max())
            te_min = pd.to_datetime(test_df[date_col]).min()
            logger.info("Vùng ngày  train: …%s | val: %s…%s | test: %s…", t_max, v_min, v_max, te_min)
            if t_max > te_min:
                warnings.append(
                    f"max(train.{date_col})={t_max} > min(test.{date_col})={te_min} → "
                    "split KHÔNG theo thời gian (hoặc ranh giới trộn lẫn). "
                    "F1 sẽ lạc quan so với thực tế — hãy chạy prepare split theo mốc thời gian."
                )
        else:
            warnings.append(f"date_col='{date_col}' không có ở cả 3 file parquet → bỏ qua kiểm tra thời gian.")
    else:
        warnings.append(
            "Chưa cấu hình date_col → KHÔNG THỂ xác minh split theo thời gian. "
            "Nếu split là ngẫu nhiên, khả năng cao có leakage tin cũ/tin mới."
        )

    # 3) Trùng lặp chéo train/test
    train_set = {_norm_text(t) for t in train_df[text_col]}
    test_set = {_norm_text(t) for t in test_df[text_col]}
    overlap = train_set & test_set
    if overlap:
        ratio = len(overlap) / max(1, len(test_set))
        warnings.append(
            f"{len(overlap)} câu trùng lặp giữa train và test ({ratio:.1%} test set) → leakage trực tiếp."
        )

    return warnings


# ---------------------------------------------------------------------------
# Dataset + collator (dynamic padding)
# ---------------------------------------------------------------------------
class FSEDataset3Class(Dataset):
    """Dataset 3 lớp: đọc parquet → (tự động) tách từ → tokenize không pad sẵn.

    __getitem__ trả về list token_id (chưa pad); Collator sẽ pad theo batch
    → mỗi batch chỉ mất tính toán đúng bằng độ dài thực tế của nó.
    """

    def __init__(
        self,
        df_or_path,
        tokenizer,
        max_len: int = 256,
        text_col: str = "text",
        label_col: str = "label",
        segmentation_mode: str = "auto",
        segmentation_cache_dir: str | Path = "data/cache/segmentation",
        min_text_chars: int = 0,
    ):
        if isinstance(df_or_path, (str, Path)):
            df = load_parquet(df_or_path, text_col, label_col, min_text_chars)
        else:
            df = df_or_path.reset_index(drop=True)

        self.labels = torch.tensor(df[label_col].astype(int).tolist(), dtype=torch.long)
        texts = df[text_col].astype(str).tolist()

        texts = ensure_segmented(texts, segmentation_mode, segmentation_cache_dir)

        # Tokenize một lần (bulk, không pad) — nhanh hơn nhiều so với tokenize từng item
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        logger.info("Dataset: %d mẫu, độ dài token trung bình %.1f, max %d.",
                    len(self), sum(map(len, self.input_ids)) / len(self),
                    max(map(len, self.input_ids)))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": int(self.labels[idx]),
        }


class Collator:
    """Pad batch theo độ dài lớn nhất trong batch (dynamic padding)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            ids = list(b["input_ids"])
            mask = list(b["attention_mask"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attention_mask.append(mask + [0] * pad)
            labels.append(b["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
