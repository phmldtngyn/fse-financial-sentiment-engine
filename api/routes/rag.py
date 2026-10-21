"""Endpoint: POST /rag/query — truy vấn ngữ cảnh top-K."""
from fastapi import APIRouter, HTTPException

from api.schemas import RAGQueryRequest, RAGResult

router = APIRouter()

_state = {}


def load():
    """Load RAG index + retriever khi startup."""
    import yaml
    from src.rag.embedder import Embedder
    from src.rag.indexer import RollingIndex
    from src.rag.retriever import Retriever

    with open("config/rag.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        index = RollingIndex.load(
            cfg["index"]["dir"],
            dim=cfg["embedder"]["dim"],
            window_days=cfg["index"]["window_days"],
        )
        embedder = Embedder(
            model_name=cfg["embedder"]["model_name"],
            max_seq_length=cfg["embedder"]["max_seq_length"],
        )
        retriever = Retriever(
            index=index,
            embedder=embedder,
            alpha=cfg["retriever"]["alpha"],
            lambda_decay=cfg["retriever"]["lambda_decay"],
        )
        _state["retriever"] = retriever
        print(f"[rag] Loaded index ({len(index)} vectors)")
    except Exception as e:
        print(f"[rag] ⚠️  Failed to load index: {e}")


@router.post("/rag/query", response_model=list[RAGResult])
def rag_query(req: RAGQueryRequest):
    if "retriever" not in _state:
        raise HTTPException(503, "RAG not loaded")

    results = _state["retriever"].query(
        req.query,
        top_m=req.top_m,
        window_hours=req.window_hours,
    )
    return [RAGResult(**r) for r in results]