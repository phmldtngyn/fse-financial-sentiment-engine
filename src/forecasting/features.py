"""Feature builder cho forecasting.

Input DataFrame (theo trade_date) phải có các cột:
    trade_date, S_t, delta_S_t, vol, return, [v_ctx_* hoặc v_ctx]

Output: X (np.ndarray), y (np.ndarray), feature_names (list)
"""
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, target: str = "log_vol",
                   scenario: str = "C") -> tuple:
    """Xây X, y theo scenario ablation.

    scenario:
        A: [vol_lag, return_lag]
        B: A + [S_t, delta_S_t]
        C: B + v_ctx
    """
    df = df.sort_values("trade_date").reset_index(drop=True).copy()

    # Lag features
    df["vol_lag"] = df["vol"].shift(1)
    df["return_lag"] = df["return"].shift(1)

    # Target
    if target == "log_vol":
        df["y"] = np.log1p(df["vol"])
    else:
        df["y"] = df["vol"]

    # Scenario features
    base = ["vol_lag", "return_lag"]
    if scenario == "A":
        feat_cols = base
    elif scenario == "B":
        feat_cols = base + ["S_t", "delta_S_t"]
    elif scenario == "C":
        v_ctx_cols = [c for c in df.columns if c.startswith("v_ctx_")]
        if not v_ctx_cols and "v_ctx" in df.columns:
            # Nếu v_ctx là cột chứa array
            v_ctx_arr = np.stack(df["v_ctx"].values)
            for i in range(v_ctx_arr.shape[1]):
                df[f"v_ctx_{i}"] = v_ctx_arr[:, i]
            v_ctx_cols = [f"v_ctx_{i}" for i in range(v_ctx_arr.shape[1])]
        feat_cols = base + ["S_t", "delta_S_t"] + v_ctx_cols
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Drop NaN (do lag 1)
    df = df.dropna(subset=["y"] + feat_cols).reset_index(drop=True)

    X = df[feat_cols].values.astype("float32")
    y = df["y"].values.astype("float32")
    return X, y, feat_cols, df["trade_date"].values