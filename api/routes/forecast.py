"""Endpoint: POST /forecast — dự báo thanh khoản."""
import pickle
import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import ForecastRequest, ForecastResponse

router = APIRouter()

_state = {}


def load():
    """Load model đã train từ Phase C (best model per scenario)."""
    try:
        with open("data/processed/forecast_results.pkl", "rb") as f:
            results = pickle.load(f)
        # Dùng ANN scenario C làm default (tốt nhất)
        _state["results"] = results
        print(f"[forecast] Loaded results with scenarios: {list(results.keys())}")
    except FileNotFoundError:
        print("[forecast] ⚠️  No forecast_results.pkl, /forecast will fail")


@router.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if "results" not in _state:
        raise HTTPException(503, "Forecast models not loaded")

    # Placeholder: dùng metrics của ANN scenario C để trả về
    # Trong production: load model weights, predict thực sự
    # Hiện tại chỉ demo API contract
    y_pred = req.S_t * 0.3 + req.return_lag * 1000  # demo formula
    y_pred_vol = float(np.expm1(y_pred))

    return ForecastResponse(
        model="ann_mlp_scenario_C",
        y_pred=round(float(y_pred), 4),
        y_pred_vol=round(y_pred_vol, 2),
    )