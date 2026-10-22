"""Kiểm tra signal S_t với volatility + check leak."""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/market_with_sentiment.parquet").dropna()

df["volatility"] = df["return"].abs()
df["volatility_5d"] = df["return"].rolling(5).std()

print("=" * 60)
print("Corr S_t với volatility")
print("=" * 60)
print(f"Corr(S_t, |return|):          {df['S_t'].corr(df['volatility']):.4f}")
print(f"Corr(S_t, rolling_std_5d):    {df['S_t'].corr(df['volatility_5d']):.4f}")
print(f"Corr(|S_t|, |return|):        {df['S_t'].abs().corr(df['volatility']):.4f}")
print(f"Corr(delta_S_t, |return|):    {df['delta_S_t'].corr(df['volatility']):.4f}")
print(f"Corr(|delta_S_t|, |return|):  {df['delta_S_t'].abs().corr(df['volatility']):.4f}")

print()
print("=" * 60)
print("Autocorrelation |return| (check leak)")
print("=" * 60)
print(f"Corr(|return_t|, |return_t-1|):   {df['volatility'].corr(df['volatility'].shift(1)):.4f}")
print(f"Corr(|return_t|, |return_t-5|):   {df['volatility'].corr(df['volatility'].shift(5)):.4f}")

print()
print("=" * 60)
print("Ngày return = 0")
print("=" * 60)
n_zero = (df["return"] == 0).sum()
print(f"Số ngày return=0: {n_zero} / {len(df)} ({n_zero/len(df)*100:.1f}%)")

print()
print("=" * 60)
print("Volatility stats")
print("=" * 60)
print(df["volatility"].describe().round(6))