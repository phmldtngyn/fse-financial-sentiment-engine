"""Recompute S_t từ model 3-class trên toàn corpus đã crawl.

Quy trình:
1. Load model 3-class
2. Score từng bài (logits → probabilities)
3. Convert → score ∈ [-1,1]: score = p_positive - p_negative
4. Weight by confidence = max(softmax)
5. Group by date → S_t, delta_S_t
"""
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from transformers import AutoTokenizer
import pandas as pd

from src.models.fse_model import FSEModel


CLASS_NAMES = ["negative", "neutral", "positive"]


def load_model(ckpt="checkpoints/fse_sft/best.pt", backbone="VinAI/phobert-base-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(backbone)
    model = FSEModel(backbone)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"✅ Model loaded: {ckpt}")
    return tok, model, device


def score_texts(tok, model, device, texts, batch_size=32):
    """Score list texts → (score, confidence, class_name) per text."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model.forward_batch(enc["input_ids"], enc["attention_mask"])

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            # score = p_positive - p_negative ∈ [-1, 1]
            score = float(p[2] - p[0])
            conf = float(p.max())
            cls = int(p.argmax())
            results.append({
                "score": score,
                "confidence": conf,
                "label": CLASS_NAMES[cls],
            })
    return results


def load_corpus():
    """Load tất cả bài từ data/parsed/*.json (crawler output)."""
    parsed_dir = Path("data/parsed")
    json_files = sorted(parsed_dir.glob("crawled_data_*.json"))
    if not json_files:
        raise FileNotFoundError(f"Không có file crawl nào trong {parsed_dir}")

    all_articles = []
    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_articles.extend(data)
        except Exception as e:
            print(f"⚠️  Skip {f.name}: {e}")

    print(f"Loaded {len(all_articles)} articles từ {len(json_files)} file")
    return all_articles


def compute_daily_sentiment(articles, scored):
    """Group by date, tính S_t = confidence-weighted mean."""
    by_date = defaultdict(list)

    for art, sc in zip(articles, scored):
        # Lấy ngày từ publish_time hoặc discovered_time
        date_str = None
        for key in ["publish_time", "discovered_time", "date"]:
            val = art.get(key, "")
            if val:
                date_str = val[:10]
                break

        if not date_str:
            continue

        by_date[date_str].append({
            "score": sc["score"],
            "conf": sc["confidence"],
            "title": art.get("title", "")[:80],
        })

    daily = []
    for date_str in sorted(by_date.keys()):
        items = by_date[date_str]
        scores = np.array([it["score"] for it in items])
        confs = np.array([it["conf"] for it in items])

        # S_t = confidence-weighted average
        S_t = float((scores * confs).sum() / confs.sum())

        daily.append({
            "date": date_str,
            "S_t": round(S_t, 4),
            "n_articles": len(items),
            "mean_conf": round(float(confs.mean()), 4),
            "n_neg": sum(1 for it in items if it["score"] < -0.2),
            "n_neu": sum(1 for it in items if -0.2 <= it["score"] <= 0.2),
            "n_pos": sum(1 for it in items if it["score"] > 0.2),
        })

    df = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    df["delta_S_t"] = df["S_t"].diff().fillna(0).round(4)
    return df


def main():
    print("=" * 60)
    print("Recompute S_t on corpus")
    print("=" * 60)

    # 1. Load model
    tok, model, device = load_model()

    # 2. Load corpus
    articles = load_corpus()
    if len(articles) == 0:
        raise ValueError("Corpus rỗng. Chạy crawler trước: python -m scripts.run_collector")

    # 3. Score
    texts = []
    for a in articles:
        # Ưu tiên content, fallback title
        t = a.get("content") or a.get("body_segmented") or a.get("title") or ""
        texts.append(t[:2000])

    print(f"\nScoring {len(texts)} articles...")
    scored = score_texts(tok, model, device, texts)

    # 4. Daily S_t
    daily_df = compute_daily_sentiment(articles, scored)

    print(f"\n✅ Computed S_t for {len(daily_df)} days")
    print(f"\nLatest 5 days:")
    print(daily_df.tail(5).to_string(index=False))

    # 5. Save
    out = Path("data/processed/sentiment_daily.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(out)
    print(f"\n✅ Saved → {out}")
    print(f"   Columns: {list(daily_df.columns)}")


if __name__ == "__main__":
    main()