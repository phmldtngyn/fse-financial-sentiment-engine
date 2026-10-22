"""Test model 3-class đã train."""
import torch
from transformers import AutoTokenizer
from src.models.fse_model import FSEModel

CLASS_NAMES = ["negative", "neutral", "positive"]

tok = AutoTokenizer.from_pretrained("VinAI/phobert-base-v2")
model = FSEModel("VinAI/phobert-base-v2")
model.load_state_dict(torch.load("checkpoints/fse_sft/best.pt", map_location="cpu", weights_only=True))
model.eval()

tests = [
    ("Cổ phiếu VNM tăng trần, khối ngoại mua ròng 200 tỷ", "positive"),
    ("Doanh nghiệp báo lỗ nặng, cổ phiếu giảm sàn", "negative"),
    ("Đại hội cổ đông thường niên diễn ra sáng nay", "neutral"),
    ("Sớm nhất TTCK VN sẽ được nâng hạng vào tháng 9", "positive"),
    ("FOMO mua đỉnh bán đáy, nhà đầu tư thua lỗ", "negative"),
]

print(f"{'Predicted':10s} {'Conf':>6s} {'Expected':10s} {'Match':6s} | Text")
print("-" * 100)

correct = 0
with torch.no_grad():
    for text, expected in tests:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        logits = model.forward_batch(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, -1)[0]
        cls = int(logits.argmax(-1).item())
        pred = CLASS_NAMES[cls]
        conf = float(probs[cls].item())
        match = "✅" if pred == expected else "❌"
        if pred == expected:
            correct += 1
        print(f"{pred:10s} {conf:>6.2f} {expected:10s} {match:6s} | {text[:60]}")

print(f"\n{correct}/5 đúng")