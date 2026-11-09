"""Test model 3-class đã train — BẢN PATCH (tích hợp model tự train).

Thay đổi so với bản cũ:
- Nạp checkpoint định dạng MỚI {"state_dict", "meta"} qua
  src.training.evaluate.load_checkpoint (bản cũ load state_dict trần
  vào FSEModel → không ăn checkpoint do train_fse_sft xuất ra).
- Tách từ underthesea trước khi tokenize (cùng chuẩn lúc train).

Thay file: scripts/test_trained_model.py
Chạy từ thư mục gốc repo:
    python -m scripts.test_trained_model
"""
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoTokenizer

from src.training.evaluate import load_checkpoint

CLASS_NAMES = ["negative", "neutral", "positive"]
CHECKPOINT = "checkpoints/fse_sft/selected/best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, meta = load_checkpoint(CHECKPOINT, device)

tok = AutoTokenizer.from_pretrained(meta.get("backbone", "VinAI/phobert-base-v2"))
from underthesea import word_tokenize  # noqa: E402  (đặt sau load model cho nhanh fail)

tests = [
    ("Cổ phiếu VNM tăng trần, khối ngoại mua ròng 200 tỷ", "positive"),
    ("Doanh nghiệp báo lỗ nặng, cổ phiếu giảm sàn", "negative"),
    ("Đại hội cổ đông thường niên diễn ra sáng nay", "neutral"),
    ("Sớm nhất TTCK VN sẽ được nâng hạng vào tháng 9", "positive"),
    ("FOMO mua đỉnh bán đáy, nhà đầu tư thua lỗ", "negative"),
]

print(f"Checkpoint: {CHECKPOINT} | seed={meta.get('seed')} | backbone={meta.get('backbone')}")
print(f"{'Predicted':10s} {'Conf':>6s} {'Expected':10s} {'Match':6s} | Text")
print("-" * 100)

correct = 0
with torch.no_grad():
    for text, expected in tests:
        seg = word_tokenize(text, format="text")
        enc = tok(seg, return_tensors="pt", truncation=True, max_length=256)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out.logits if hasattr(out, "logits") else out
        probs = torch.softmax(logits, -1)[0]
        cls = int(logits.argmax(-1).item())
        pred = CLASS_NAMES[cls]
        conf = float(probs[cls].item())
        match = "OK" if pred == expected else "SAI"
        if pred == expected:
            correct += 1
        print(f"{pred:10s} {conf:>6.2f} {expected:10s} {match:6s} | {text[:60]}")

print(f"\n{correct}/5 đúng")
