"""Endpoint: GET /metrics — trả về metrics ablation."""
import pickle
from fastapi import APIRouter, HTTPException

router = APIRouter()

_state = {}


def load():
    try:
        with open("data/processed/forecast_results.pkl", "rb") as f:
            _state["results"] = pickle.load(f)
        print("[metrics] Loaded forecast_results.pkl")
    except FileNotFoundError:
        print("[metrics] ⚠️  No forecast_results.pkl")


@router.get("/metrics")
def get_metrics():
    if "results" not in _state:
        raise HTTPException(503, "Metrics not loaded")

    out = {}
    for scenario, models in _state["results"].items():
        out[scenario] = {}
        for model, data in models.items():
            out[scenario][model] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in data["metrics"].items()
            }
    return out