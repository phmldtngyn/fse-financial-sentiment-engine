FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    default-jre-headless \
    libglib2.0-0 \
    libnss3 \
    libfontconfig1 \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .

# Cài đặt PyTorch bản CPU siêu nhẹ trước, sau đó cài các dependencies khác
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium && \
    playwright install-deps chromium

RUN mkdir -p /app/data/raw \
    /app/data/parsed \
    /app/data/processed \
    /app/data/gold_standard \
    /app/data/embeddings \
    /app/logs

COPY . /app

CMD ["python", "scripts/run_daily_pipeline.py"]