"""Sinh dummy market data (500 ngày) để test pipeline forecasting.

Thiết kế signal rõ ràng để verify scenario C > B > A:
- vol_t phụ thuộc TUYẾN TÍNH vào S_t (LR học được)
- vol_t phụ thuộc PHI TUYẾN vào |S_t| (DT/kNN học được)
- v_ctx 4 chiều, CẢ 4 đều có signal ĐỘC LẬP với S_t (scenario C cải thiện)
- return_t có signal yếu

Usage: python -m scripts.create_dummy_market
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

N_DAYS = 500
V_CTX_DIM = 4  # giảm từ 8 → 4, tất cả đều có signal
SEED = 42

np.random.seed(SEED)


def main():
    # S_t: AR(1) có autocorrelation nhẹ
    S_t = np.zeros(N_DAYS)
    for i in range(1, N_DAYS):
        S_t[i] = 0.7 * S_t[i - 1] + 0.3 * np.random.randn()
    S_t = np.clip(S_t, -1, 1)

    # v_ctx: 4 chiều, CẢ 4 đều có signal ĐỘC LẬP với S_t
    # (khác với version cũ: v_ctx_0 collinear với S_t)
    v_ctx = np.random.randn(N_DAYS, V_CTX_DIM) * 0.6
    # v_ctx_0: tương quan âm với S_t (độc lập nhưng có ý nghĩa)
    v_ctx[:, 0] = -0.3 * S_t + np.random.randn(N_DAYS) * 0.5
    # v_ctx_1: momentum của return (không từ S_t)
    return_raw = np.random.randn(N_DAYS) * 0.02
    v_ctx[:, 1] = np.concatenate([[0], return_raw[:-1]]) + np.random.randn(N_DAYS) * 0.01
    # v_ctx_2, v_ctx_3: noise nhưng có ảnh hưởng nhỏ đến vol (giữ làm signal yếu)
    # → giữ nguyên random

    # Return có tương quan âm nhẹ với S_t
    return_t = -0.1 * S_t + return_raw

    # Vol: phụ thuộc cả TUYẾN TÍNH (S_t) và PHI TUYẾN (|S_t|), + v_ctx signal
    vol_t = (
        1e6
        + 5e5 * np.abs(return_t)          # |return| (phi tuyến)
        + 2e5 * S_t                       # tuyến tính (LR học được)
        + 3e5 * np.abs(S_t)               # |S_t| (DT/kNN học được)
        + 1.5e5 * np.abs(v_ctx[:, 0])     # v_ctx_0 signal
        + 1e5 * np.abs(v_ctx[:, 1])       # v_ctx_1 signal
        + 5e4 * np.abs(v_ctx[:, 2])       # v_ctx_2 signal yếu
        + 5e4 * np.abs(v_ctx[:, 3])       # v_ctx_3 signal yếu
        + np.random.rand(N_DAYS) * 3e4    # noise nhỏ
    ).astype("float32")

    delta_S_t = np.concatenate([[0], np.diff(S_t)])

    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")

    df = pd.DataFrame({
        "trade_date": dates,
        "S_t": S_t.astype("float32"),
        "delta_S_t": delta_S_t.astype("float32"),
        "return": return_t.astype("float32"),
        "vol": vol_t,
    })
    for i in range(V_CTX_DIM):
        df[f"v_ctx_{i}"] = v_ctx[:, i].astype("float32")

    df.to_parquet(OUT / "dummy_market.parquet")
    print(f"✅ Generated {len(df)} days → {OUT}/dummy_market.parquet")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   S_t: mean={S_t.mean():.3f}, std={S_t.std():.3f}")
    print(f"   vol: mean={vol_t.mean():.0f}, std={vol_t.std():.0f}")
    print(f"\n   💡 Signal thiết kế:")
    print(f"   - S_t (tuyến tính)       → LR học được")
    print(f"   - |S_t| (phi tuyến)      → DT/kNN học được")
    print(f"   - v_ctx (4 chiều, độc lập) → scenario C cải thiện")
    print(f"   → Kỳ vọng: C > B > A ở hầu hết models")


if __name__ == "__main__":
    main()