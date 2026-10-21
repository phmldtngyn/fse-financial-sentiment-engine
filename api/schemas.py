"""Pydantic schemas cho API."""
from typing import List
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Bài báo/câu cần chấm cảm xúc")


class SentimentResponse(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0, description="Điểm cảm xúc ∈ [-1,1]")
    tier: int = Field(..., ge=1, le=3, description="Mức độ liên quan ∈ {1,2,3}")


class RAGQueryRequest(BaseModel):
    query: str
    top_m: int = 5
    window_hours: int = 168


class RAGResult(BaseModel):
    id: int
    text: str
    published_at: str
    sim: float
    age_hours: float
    score: float


class ForecastRequest(BaseModel):
    S_t: float
    delta_S_t: float
    vol_lag: float
    return_lag: float
    v_ctx: List[float] = Field(default_factory=list)


class ForecastResponse(BaseModel):
    model: str
    y_pred: float
    y_pred_vol: float = Field(..., description="Dự báo thanh khoản (đã inverse log)")


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict