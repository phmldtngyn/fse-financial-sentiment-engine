"""FastAPI app — FSE Engine API.

Chạy: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs

Endpoints:
    POST /api/v1/sentiment    — chấm điểm cảm xúc tin tức (3-class)
    POST /api/v1/rag/query    — truy vấn ngữ cảnh RAG
    GET  /api/v1/correlation  — kết quả phân tích tương quan banking-sentiment
    GET  /health              — health check
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import sentiment, rag, correlation
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models + data khi startup."""
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    sentiment.load()
    rag.load()
    correlation.load()

    print("=" * 60)
    print("All modules loaded. API ready.")
    print("=" * 60)
    yield
    print("Shutting down...")


app = FastAPI(
    title="FSE Engine API",
    description=(
        "Financial Sentiment Engine — "
        "3-class sentiment classification + RAG context + "
        "banking-sentiment correlation analysis"
    ),
    version="2.0.0",
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
app.include_router(correlation.router, prefix="/api/v1", tags=["correlation"])


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded={
            "sentiment": "model" in sentiment._state,
            "rag": "retriever" in rag._state,
            "correlation": "results" in correlation._state,
        },
    )