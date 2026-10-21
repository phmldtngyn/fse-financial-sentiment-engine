"""Endpoint: POST /sentiment — chấm điểm cảm xúc từ text."""
import torch
from fastapi import APIRouter, HTTPException
from transformers import AutoTokenizer

from src.models.fse_model import FSEModel
from api.schemas import SentimentRequest, SentimentResponse

router = APIRouter()

_state = {}


def load():
    """Load model 1 lần khi startup."""
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
        nss_logit, tier_logits = model.forward_batch(
            enc["input_ids"], enc["attention_mask"],
        )

    score = float(torch.tanh(nss_logit).cpu())
    tier = int(tier_logits.argmax(-1).cpu()) + 1

    return SentimentResponse(score=round(score, 4), tier=tier)