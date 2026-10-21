"""Tạo lại 6 file Docker bị hỏng here-string."""
from pathlib import Path

ROOT = Path(".")

FILES = {
    ".dockerignore": """__pycache__/
*.py[cod]
*.so
.Python
venv/
.venv/
env/
ENV/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.env
*.env
.env.*
.vscode/
.idea/
*.swp
*.swo
.git/
.gitignore
data/raw/
data/parsed/
data/gold/
data/processed/*.json
data/processed/*.parquet
data/embeddings/
checkpoints/
models/checkpoints/
.cache/
.hf_cache/
*.log
logs/
""",

    "requirements-api.txt": """torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.0.0
pandas>=2.0.0
pyarrow>=14.0.0
PyYAML>=6.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
""",

    "requirements-dashboard.txt": """streamlit>=1.40.0
httpx>=0.27.0
pandas>=2.0.0
""",

    "Dockerfile.api": """FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    HF_HOME=/app/.hf_cache \\
    TRANSFORMERS_CACHE=/app/.hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential \\
        libgomp1 \\
        curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --upgrade pip && \\
    pip install -r requirements-api.txt

COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY constants.py .

RUN mkdir -p checkpoints/fse_sft data/rag_index data/processed .hf_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",

    "Dockerfile.dashboard": """FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements-dashboard.txt .

RUN pip install --upgrade pip && \\
    pip install -r requirements-dashboard.txt

COPY dashboard/ ./dashboard/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \\
     "--server.port=8501", \\
     "--server.address=0.0.0.0", \\
     "--server.headless=true", \\
     "--browser.gatherUsageStats=false"]
""",

    "docker-compose.yml": """services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: fse-api
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./data/rag_index:/app/data/rag_index
      - ./data/processed:/app/data/processed
      - ./.hf_cache:/app/.hf_cache
    environment:
      - HF_HOME=/app/.hf_cache
      - TRANSFORMERS_CACHE=/app/.hf_cache
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: fse-dashboard
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
""",
}


def main():
    for filename, content in FILES.items():
        path = ROOT / filename
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"✅ Wrote {filename} ({len(content)} chars)")

    print("\n🎉 Done. All 6 files recreated with clean content.")


if __name__ == "__main__":
    main()