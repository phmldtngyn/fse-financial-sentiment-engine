"""Train 4 forecasting models với walk-forward CV + ablation.

Usage:
    python -m scripts.train_forecast                    # dùng dummy
    python -m scripts.train_forecast --data path.parquet
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.forecasting.ablation import run_full_ablation, print_summary, MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dummy_market.parquet")
    parser.add_argument("--out", default="data/processed/forecast_results.pkl")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Không thấy {data_path}. Chạy create_dummy_market trước.")

    print(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    results = run_full_ablation(df, verbose=True)

    print_summary(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✅ Results saved → {out_path}")


if __name__ == "__main__":
    main()