"""Lấy 2.000 câu từ ViFiD → chia 20 batch × 100 câu để gán nhãn."""
from pathlib import Path
import pandas as pd

VIFID = Path(r"C:\Users\ADMIN\.cache\kagglehub\datasets\daddychillonkaggle\vifid-segraw-200k\versions\3\ViFiD-200K.csv")
OUT = Path("data/gold")
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 2000
BATCH_SIZE = 100
SEED = 42

def main():
    if not VIFID.exists():
        raise FileNotFoundError(f"Không thấy ViFiD tại {VIFID}")

    df = pd.read_csv(VIFID)[["label"]].rename(columns={"label": "text"})
    df = df[df["text"].astype(str).str.len().between(40, 800)]
    df = df.drop_duplicates("text").reset_index(drop=True)
    print(f"Câu hợp lệ sau lọc: {len(df):,}")

    sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)
    sample.insert(0, "id", range(1, len(sample) + 1))

    n_batches = 0
    for i in range(0, len(sample), BATCH_SIZE):
        chunk = sample.iloc[i:i + BATCH_SIZE]
        fp = OUT / f"batch_{i // BATCH_SIZE + 1:02d}.csv"
        chunk.to_csv(fp, index=False, encoding="utf-8")
        n_batches += 1

    # Cũng lưu file tổng để merge sau
    sample.to_csv(OUT / "all_sampled.csv", index=False, encoding="utf-8")

    print(f"\n✅ Đã chia {len(sample):,} câu → {n_batches} batch × {BATCH_SIZE}")
    print(f"   {OUT}/batch_01.csv … batch_{n_batches:02d}.csv")
    print(f"   Tổng: {OUT}/all_sampled.csv")

if __name__ == "__main__":
    main()