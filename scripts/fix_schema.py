"""Chuẩn hóa schema: gộp Title + Content → text."""
import pandas as pd
from pathlib import Path

for split in ["train", "val", "test"]:
    p = Path(f"data/processed/{split}.parquet")
    if not p.exists():
        print(f"⚠️  {p.name} missing")
        continue

    df = pd.read_parquet(p)

    if "text" not in df.columns:
        title = df["Title"].fillna("").astype(str).str.strip() if "Title" in df.columns else ""
        content = df["Content"].fillna("").astype(str).str.strip() if "Content" in df.columns else ""
        df["text"] = (title + " . " + content).str.strip()
        df = df[["text", "score", "tier"]]
        df.to_parquet(p)

    print(f"✅ {split}: {len(df):,} rows | Cols: {list(df.columns)}")