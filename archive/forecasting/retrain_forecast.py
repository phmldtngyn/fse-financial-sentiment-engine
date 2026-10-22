"""Retrain 4 forecast models trên dataset S_t (Sentiment_NLP proxy)."""
import pickle
from pathlib import Path
import pandas as pd

from src.forecasting.ablation import run_full_ablation, print_summary


def main():
    data_path = Path("data/processed/market_with_sentiment.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Chạy build_forecast_dataset.py trước. Thiếu {data_path}")

    print(f"Loading: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  {len(df)} days")
    print(f"  Date range: {df['trade_date'].min()} → {df['trade_date'].max()}")
    print(f"  S_t non-zero: {(df['S_t'] != 0).sum():,} / {len(df):,} "
          f"({(df['S_t'] != 0).mean()*100:.1f}%)")

    if len(df) < 100:
        print(f"\n⚠️  Chỉ {len(df)} ngày — không đủ train.")
        return

    results = run_full_ablation(df, verbose=True)
    print_summary(results)

    out_path = Path("data/processed/forecast_results_real.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✅ Saved → {out_path}")
    print(f"\n📊 Copy sang pkl chính:")
    print(f"   Copy-Item data\\processed\\forecast_results_real.pkl data\\processed\\forecast_results.pkl -Force")


if __name__ == "__main__":
    main()