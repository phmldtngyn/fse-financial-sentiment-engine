"""Sinh dummy gold standard (500 câu) để test pipeline SFT không cần Claude.

Sau khi test xong → xóa file này, chạy finalize_gold.py với data thật từ Claude.

Usage: python -m scripts.create_dummy_gold
"""
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

N_TOTAL = 500
SEED = 42
random.seed(SEED)

POSITIVE = [
    "Cổ phiếu tăng trần với khối lượng giao dịch đột biến",
    "Doanh nghiệp báo lãi kỷ lục quý III",
    "Khối ngoại mua ròng hơn 500 tỷ đồng",
    "Cổ phiếu VNM tăng 5% sau kết quả kinh doanh khả quan",
    "Ngân hàng tăng trưởng tín dụng 20% so với cùng kỳ",
    "Lợi nhuận sau thuế tăng 45%, vượt kế hoạch năm",
    "Cổ đông lớn đăng ký mua thêm 2 triệu cổ phiếu",
]
NEGATIVE = [
    "Cổ phiếu giảm sàn không có lối thoát",
    "Doanh nghiệp báo lỗ lớn quý III",
    "Khối ngoại bán ròng gần 1.000 tỷ đồng",
    "Cổ phiếu HPG lao dốc 7% sau tin xấu",
    "Nợ xấu ngân hàng tăng vọt lên 3%",
    "Công ty bị phong tỏa tài khoản, cổ phiếu mất thanh khoản",
    "Lãnh đạo bị khởi tố, cổ phiếu giảm 15%",
]
NEUTRAL = [
    "Đại hội cổ đông thường niên diễn ra sáng nay",
    "Thị trường đi ngang, thanh khoản thấp",
    "NHNN giữ nguyên lãi suất điều hành",
    "Cổ phiếu VCB công bố lịch chốt quyền cổ tức",
    "Báo cáo tài chính quý III được công bố",
    "Chỉ số VN-Index đóng cửa tại 1.250 điểm",
    "Ủy ban Chứng khoán công bố thông tư mới",
]


def make_text(kind: str) -> str:
    base = random.choice({"pos": POSITIVE, "neg": NEGATIVE, "neu": NEUTRAL}[kind])
    return base


def main():
    rows = []
    for _ in range(N_TOTAL):
        kind = random.choice(["pos", "neg", "neu"])
        score = {
            "pos": random.uniform(0.3, 1.0),
            "neg": random.uniform(-1.0, -0.3),
            "neu": random.uniform(-0.2, 0.2),
        }[kind]
        tier = (
            random.choice([2, 3]) if kind != "neu"
            else random.choice([1, 2])
        )
        rows.append({
            "text": make_text(kind),
            "score": round(score, 3),
            "tier": tier,
        })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} dummy samples")
    print(f"\nScore distribution:\n{df['score'].describe()}")
    print(f"\nTier distribution:\n{df['tier'].value_counts().sort_index()}")

    train, rest = train_test_split(df, test_size=0.2, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED)

    train.to_parquet(OUT / "train.parquet")
    val.to_parquet(OUT / "val.parquet")
    test.to_parquet(OUT / "test.parquet")

    print(f"\n✅ DUMMY GOLD (chỉ để test pipeline, KHÔNG dùng train thật):")
    print(f"   train={len(train)} val={len(val)} test={len(test)}")
    print(f"   → {OUT}/")


if __name__ == "__main__":
    main()