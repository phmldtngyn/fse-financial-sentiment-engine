"""FastAPI app — FSE Engine API.

Chạy: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import sentiment, rag, forecast, metrics
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tất cả models khi startup."""
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    sentiment.load()
    rag.load()
    forecast.load()
    metrics.load()

    print("=" * 60)
    print("All models loaded. API ready.")
    print("=" * 60)
    yield
    print("Shutting down...")


app = FastAPI(
    title="FSE Engine API",
    description="Financial Sentiment Engine — sentiment + RAG + liquidity forecast",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentiment.router, prefix="/api/v1", tags=["sentiment"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
app.include_router(forecast.router, prefix="/api/v1", tags=["forecast"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded={
            "sentiment": "model" in sentiment._state,
            "rag": "retriever" in rag._state,
            "forecast": "results" in forecast._state,
            "metrics": "results" in metrics._state,
        },
    )