"""Endpoint: GET /correlation — trả kết quả correlation analysis."""
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

_state = {}


def load():
    path = Path("data/processed/correlation/correlation_results.csv")
    if not path.exists():
        print(f"[correlation] ⚠️  No results at {path}")
        return
    _state["results"] = pd.read_csv(path).to_dict(orient="records")
    print(f"[correlation] Loaded {len(_state['results'])} pairs")


@router.get("/correlation")
def get_correlation():
    if "results" not in _state:
        raise HTTPException(503, "Correlation results not loaded. Run scripts/correlation_analysis.py first.")
    return _state["results"]