"""Endpoint: POST /sentiment — 3-class sentiment classification."""
import torch
from fastapi import APIRouter, HTTPException
from transformers import AutoTokenizer

from src.models.fse_model import FSEModel
from api.schemas import SentimentRequest, SentimentResponse

router = APIRouter()

_state = {}

# Class 0=negative, 1=neutral, 2=positive
CLASS_NAMES = ["negative", "neutral", "positive"]
# Map class → pseudo-score để tương thích dashboard cũ
CLASS_TO_SCORE = {0: -0.7, 1: 0.0, 2: 0.7}


def load():
    ckpt_path = "checkpoints/fse_sft/best.pt"
    backbone = "VinAI/phobert-base-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(backbone)
    model = FSEModel(backbone)
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[sentiment] Loaded SFT checkpoint: {ckpt_path}")
    except FileNotFoundError:
        print(f"[sentiment] ⚠️  No checkpoint at {ckpt_path}, using random heads")
    except RuntimeError as e:
        print(f"[sentiment] ⚠️  Checkpoint incompatible (old regression format): {e}")
        print(f"[sentiment]    → Using random head. Train 3-class model to fix.")
    model.to(device).eval()

    _state["tok"] = tok
    _state["model"] = model
    _state["device"] = device


@router.post("/sentiment", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    if "model" not in _state:
        raise HTTPException(503, "Model not loaded")

    tok = _state["tok"]
    model = _state["model"]
    device = _state["device"]

    enc = tok(
        req.text, truncation=True, max_length=256,
        padding="max_length", return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model.forward_batch(enc["input_ids"], enc["attention_mask"])

    probs = torch.softmax(logits, dim=-1)[0]
    cls = int(logits.argmax(-1).item())
    confidence = float(probs[cls].item())
    score = CLASS_TO_SCORE[cls]

    # Tier giữ = 3 (không còn tier head) để dashboard không phải sửa
    return SentimentResponse(score=score, tier=3)