"""Diagnose S_t distribution trong forecast dataset."""
import pandas as pd

df = pd.read_parquet("data/processed/market_with_sentiment.parquet")

print("=" * 60)
print("S_t distribution")
print("=" * 60)
print(df["S_t"].describe().round(4))

print(f"\nSố ngày S_t = 0:       {(df['S_t'] == 0).sum():,} ({(df['S_t'] == 0).mean()*100:.1f}%)")
print(f"Số ngày S_t > 0:       {(df['S_t'] > 0).sum():,} ({(df['S_t'] > 0).mean()*100:.1f}%)")
print(f"Số ngày S_t < 0:       {(df['S_t'] < 0).sum():,} ({(df['S_t'] < 0).mean()*100:.1f}%)")

print(f"\nS_t theo năm:")
df["year"] = pd.to_datetime(df["date"]).dt.year
yearly = df.groupby("year").agg(
    n=("S_t", "size"),
    n_nonzero=("S_t", lambda x: (x != 0).sum()),
    mean=("S_t", "mean"),
    std=("S_t", "std"),
).round(3)
yearly["pct_nonzero"] = (yearly["n_nonzero"] / yearly["n"] * 100).round(1)
print(yearly.to_string())

print(f"\nTop 10 ngày có |S_t| lớn nhất:")
top = df.nlargest(10, "S_t")[["date", "S_t", "vol"]]
print(top.to_string(index=False))