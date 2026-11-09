"""Endpoint: POST /sentiment — 3-class sentiment classification.

BẢN PATCH (tích hợp model tự train trên Colab):
- Nạp checkpoint định dạng MỚI {"state_dict", "meta"} qua
  src.training.evaluate.load_checkpoint — CÙNG một đường nạp với training/eval,
  hết tình trạng API dùng FSEModel riêng nên không ăn được checkpoint mới.
- Nếu chưa có checkpoint: POST /sentiment trả 503 với hướng dẫn rõ
  (không còn âm thầm chạy random head như bản cũ).
- Text được tách từ bằng underthesea trước khi tokenize (model train trên
  text đã tách từ ở segmentation mode auto — serving phải dùng cùng chuẩn).

Thay file: api/routes/sentiment.py
Yêu cầu thêm cho môi trường API: pip install underthesea
"""
import torch
from fastapi import APIRouter, HTTPException

from api.schemas import SentimentRequest, SentimentResponse

router = APIRouter()

_state = {}

# Class 0=negative, 1=neutral, 2=positive → pseudo-score tương thích dashboard cũ
CLASS_TO_SCORE = {0: -0.7, 1: 0.0, 2: 0.7}
CHECKPOINT_CANDIDATES = [
    "checkpoints/fse_sft/selected/best.pt",   # output pipeline mới (train_fse_sft)
    "checkpoints/fse_sft/best.pt",            # vị trí cũ (nếu còn)
]


def load():
    from src.training.evaluate import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for path in CHECKPOINT_CANDIDATES:
        try:
            model, meta = load_checkpoint(path, device)
        except FileNotFoundError:
            continue
        _state["model"] = model
        _state["meta"] = meta
        _state["device"] = device
        print(f"[sentiment] Loaded checkpoint: {path} (seed={meta.get('seed')})")
        return
    _state["model"] = None
    print(f"[sentiment] Không tìm thấy checkpoint ở {CHECKPOINT_CANDIDATES} "
          f"→ POST /sentiment sẽ trả 503. Hãy đặt best.pt của Colab vào "
          f"checkpoints/fse_sft/selected/best.pt")


@router.post("/sentiment", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    if not _state.get("model"):
        raise HTTPException(
            503,
            "Model not loaded — hãy train rồi đặt checkpoint vào "
            "checkpoints/fse_sft/selected/best.pt",
        )

    from transformers import AutoTokenizer
    from underthesea import word_tokenize

    model, device = _state["model"], _state["device"]

    tok = _state.get("tok")
    if tok is None:
        backbone = _state.get("meta", {}).get("backbone", "VinAI/phobert-base-v2")
        tok = AutoTokenizer.from_pretrained(backbone)
        _state["tok"] = tok

    # Model train trên text đã tách từ → serving cũng phải tách từ
    text_seg = word_tokenize(req.text, format="text")

    enc = tok(text_seg, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        logits = out.logits if hasattr(out, "logits") else out

    cls = int(logits.argmax(-1).item())
    score = CLASS_TO_SCORE[cls]

    # Tier giữ = 3 (không còn tier head) để dashboard không phải sửa
    return SentimentResponse(score=score, tier=3)
