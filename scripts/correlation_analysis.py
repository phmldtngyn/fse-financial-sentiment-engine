"""Correlation analysis: S_t vs banking volume/return/range."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/processed/banking_with_sentiment.parquet")
OUT_DIR = Path("data/processed/correlation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def corr_test(x, y, method="pearson"):
    mask = x.notna() & y.notna()
    x_, y_ = x[mask], y[mask]
    if len(x_) < 10:
        return np.nan, np.nan, len(x_)
    if method == "pearson":
        r, p = stats.pearsonr(x_, y_)
    else:
        r, p = stats.spearmanr(x_, y_)
    return round(r, 4), round(p, 6), len(x_)


def lag_correlation(x, y, max_lag=5):
    results = []
    for lag in range(max_lag + 1):
        r, p, n = corr_test(x.shift(lag), y) if lag > 0 else corr_test(x, y)
        results.append({"lag": lag, "r": r, "p_value": p, "n": n})
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Correlation Analysis: S_t vs Banking")
    print("=" * 60)

    df = pd.read_parquet(DATA).sort_values("trade_date").reset_index(drop=True)
    print(f"Loaded {len(df)} days ({df['trade_date'].min()} → {df['trade_date'].max()})")

    pairs = [
        ("S_t", "total_volume", "S_t vs Volume"),
        ("S_t", "log_volume", "S_t vs log(Volume)"),
        ("S_t", "avg_return", "S_t vs Return"),
        ("S_t", "avg_range", "S_t vs Intraday Range"),
        ("S_t", "abs_return", "S_t vs |Return|"),
        ("abs_S_t", "avg_range", "|S_t| vs Intraday Range"),
        ("abs_S_t", "log_volume", "|S_t| vs log(Volume)"),
        ("abs_S_t", "abs_return", "|S_t| vs |Return|"),
    ]

    rows = []
    print(f"\n{'='*72}")
    print(f"{'Pair':<35} {'Pearson':>10} {'p-val':>10} {'Spearman':>10} {'n':>6}")
    print('='*72)

    for x_col, y_col, label in pairs:
        r_p, p_p, n = corr_test(df[x_col], df[y_col], "pearson")
        r_s, p_s, _ = corr_test(df[x_col], df[y_col], "spearman")
        sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else ""
        print(f"{label:<35} {r_p:>10.4f} {p_p:>10.6f} {r_s:>10.4f} {n:>6}{sig}")
        rows.append({
            "pair": label, "x": x_col, "y": y_col,
            "pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s,
            "n": n, "significant": p_p < 0.05,
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "correlation_results.csv", index=False)
    print(f"\n✅ Saved {OUT_DIR/'correlation_results.csv'}")

    print(f"\n{'='*72}")
    print("Lag correlation (S_t shifted vs target today)")
    print('='*72)

    lag_frames = []
    for target, label in [("log_volume", "Volume"), ("avg_return", "Return"), ("avg_range", "Range")]:
        lag_df = lag_correlation(df["S_t"], df[target], max_lag=5)
        lag_df["target"] = label
        lag_frames.append(lag_df)
        print(f"\nS_t(lag) vs {label}:")
        print(lag_df.to_string(index=False))

    pd.concat(lag_frames).to_csv(OUT_DIR / "correlation_lag.csv", index=False)
    print(f"\n✅ Saved {OUT_DIR/'correlation_lag.csv'}")

    for x_col, y_col, title in [
        ("S_t", "log_volume", "S_t vs log(Volume)"),
        ("S_t", "avg_return", "S_t vs Return"),
        ("S_t", "avg_range", "S_t vs Range"),
        ("abs_S_t", "avg_range", "abs_S_t vs Range"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = df[x_col].notna() & df[y_col].notna()
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], alpha=0.3, s=10)
        r, p, _ = corr_test(df[x_col], df[y_col])
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{title}\nr={r:.4f}, p={p:.4f}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = title.replace("|", "").replace("(", "").replace(")", "").replace(" ", "_")
        fig.savefig(OUT_DIR / f"scatter_{safe}.png", dpi=100)
        plt.close(fig)

    rolling = pd.DataFrame({
        "trade_date": df["trade_date"],
        "r_volume": df["S_t"].rolling(60).corr(df["log_volume"]),
        "r_return": df["S_t"].rolling(60).corr(df["avg_return"]),
        "r_range": df["S_t"].rolling(60).corr(df["avg_range"]),
    })
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(rolling["trade_date"]), rolling["r_volume"], label="vs Volume")
    ax.plot(pd.to_datetime(rolling["trade_date"]), rolling["r_return"], label="vs Return")
    ax.plot(pd.to_datetime(rolling["trade_date"]), rolling["r_range"], label="vs Range")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title("Rolling 60-day Correlation")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rolling_correlation.png", dpi=100)
    plt.close(fig)
    print(f"\n✅ Saved 5 PNG plots → {OUT_DIR}/")

    sig = results_df[results_df["significant"]]
    print(f"\n{'='*72}")
    print(f"Significant pairs (p<0.05): {len(sig)}/{len(results_df)}")
    print('='*72)
    if len(sig) > 0:
        print(sig[["pair", "pearson_r", "pearson_p", "spearman_r"]].to_string(index=False))


if __name__ == "__main__":
    main()