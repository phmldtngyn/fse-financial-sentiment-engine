"""
constants.py - System Constants and Baseline Parameters for FSE
Mô hình Phân tích Cảm xúc Tin tức Tài chính và Định lượng Chỉ số Cảm xúc Dòng tiền Thị trường
"""

from typing import Dict, List, Final

# ==============================================================================
# 1. DANH SÁCH 27 NGÂN HÀNG NIÊM YẾT (Chốt ngày 22/09/2026) [Phụ lục A]
# ==============================================================================
TARGET_BANK_TICKERS: Final[List[str]] = [
    # HOSE (20 mã)
    "ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "SHB", "SSB",
    "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "OCB", "NAB", "BVB", "VBB",
    # HNX (02 mã)
    "BAB", "NVB",
    # UPCoM (05 mã)
    "ABB", "KLB", "PGB", "SGB", "VAB"
]

BANK_TICKER_TO_EXCHANGE: Final[Dict[str, str]] = {
    **{ticker: "HOSE" for ticker in [
        "ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "SHB", "SSB",
        "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "OCB", "NAB", "BVB", "VBB"
    ]},
    **{ticker: "HNX" for ticker in ["BAB", "NVB"]},
    **{ticker: "UPCOM" for ticker in ["ABB", "KLB", "PGB", "SGB", "VAB"]}
}

# Biên độ giao dịch theo sàn (%) [Mục VI.4]
EXCHANGE_TRADING_BANDS: Final[Dict[str, float]] = {
    "HOSE": 0.07,   # ±7%
    "HNX": 0.10,    # ±10%
    "UPCOM": 0.15   # ±15%
}


# ==============================================================================
# 2. CAU HINH NGUON TIN VA UY TIN NGUON (SourceCredibility) [Mục I.1 & VI.1.2]
# ==============================================================================
FINANCIAL_NEWS_SOURCES: Final[List[str]] = [
    "Thời báo Tài chính Việt Nam",
    "Vietnam Finance",
    "CafeF",
    "Vietstock"
]

# Baseline W_source khoi tao cho 04 nguon tin [Muc VIII.2]
BASELINE_W_SOURCE: Final[Dict[str, float]] = {
    "Thời báo Tài chính Việt Nam": 1.0,  # Bien tren Nhom 1
    "Vietnam Finance": 1.0,               # Bien tren Nhom 1
    "CafeF": 0.8,                         # Giua Nhom 2
    "Vietstock": 0.8                      # Giua Nhom 2
}

# Khoang dao dong tinh chinh W_source khi chay Optimization
W_SOURCE_BOUNDS: Final[Dict[str, tuple]] = {
    source: (max(0.0, weight - 0.2), min(1.0, weight + 0.2))
    for source, weight in BASELINE_W_SOURCE.items()
}


# ==============================================================================
# 3. TRONG SO VI TRI (PositionWeight) & TIER PHAN TANG [Mục III.3.2 & VI.1.1]
# ==============================================================================
# Trong so vi tri hien thi tren HTML (Position Tier P1 - P6) [Muc VI.1.1]
POSITION_TIER_WEIGHTS: Final[Dict[str, float]] = {
    "P1": 1.0,  # Trang chu - bai noi bat (Hero/Featured)
    "P2": 0.8,  # Trang chu - bai trong danh sach chinh
    "P3": 0.6,  # Trang chuyen muc - bai noi bat
    "P4": 0.4,  # Trang chuyen muc - bai trong danh sach (Default)
    "P5": 0.2,  # Chi xuat hien o trang chi tiet
    "P6": 0.1   # Bai lien quan / Bai cu dao lai
}

DEFAULT_POSITION_TIER: Final[str] = "P4"

# Thang phan tang do lien quan (Relevance Tier 1 - 4) [Muc III.3.2 & VIII.2]
RELEVANCE_TIER_BASELINE_WEIGHTS: Final[Dict[int, float]] = {
    1: 1.0,  # Tier 1: Lien quan truc tiep den ngan hang/ma co phieu target
    2: 0.7,  # Tier 2: Lien quan den nganh tai chinh - ngan hang
    3: 0.3,  # Tier 3: Lien quan den kinh te vi mo
    4: 0.0   # Tier 4: Khong lien quan / tin le duong
}


# ==============================================================================
# 4. SIEU THAM SO KINH TE LUONG & TAI CHINH HANH VI (Vector Theta) [Mục VIII.2]
# ==============================================================================
# Vector Theta baseline khởi tạo
BASELINE_THETA: Final[Dict[str, float]] = {
    "gamma": 1.75,  # Prospect Theory - Phóng đại tác động tiêu cực [Mục VI.3.1]
    "lambda_decay": 0.60,  # Memory Decay - Quán tính tâm lý S_t-1 [Mục VI.3.2]
    "mu": 0.05,     # Time Decay - Tốc độ lãng quên tin tức theo giờ [Mục VI.2.2]
    "alpha": 0.5    # Credibility - Influence Balance [Mục VI.1.2]
}

# Miền không gian tối ưu hóa (Optimization Bounds for Optuna / Grid Search)
OPTIMIZATION_BOUNDS: Final[Dict[str, tuple]] = {
    "gamma": (1.0, 3.0),
    "lambda_decay": (0.0, 1.0),
    "mu": (0.01, 0.2),
    "alpha": (0.0, 1.0)
}

# Tham so Novelty Score phi tuyến (κ = 2.0, Cửa sổ Δt = 48 giờ) [Mục V.2 & V.3]
SIMILARITY_LOOKBACK_HOURS: Final[int] = 48
NOVELTY_KAPPA: Final[float] = 2.0
NOVELTY_SIM_HIGH_THRESHOLD: Final[float] = 0.85
NOVELTY_SIM_LOW_THRESHOLD: Final[float] = 0.30

# Giờ giao dịch trong ngày trên sàn HOSE (4.5 giờ/ngày) [Mục VI.2.2]
DAILY_TRADING_HOURS: Final[float] = 4.5


# ==============================================================================
# 5. CAU HINH KHUNG MÔ HÌNH FSE & HUAN LUYEN NLP [Mục III & VIII.1]
# ==============================================================================
MODEL_BACKBONE_NAME: Final[str] = "VinAI/phobert-base-v2"
EMBEDDING_DIM: Final[int] = 768
MAX_SEQUENCE_LENGTH: Final[int] = 512
ATTENTION_HEADS: Final[int] = 4

# Siêu tham số Huấn luyện FSE [Mục VIII.1]
TRAINING_HYPERPARAMS: Final[Dict[str, float]] = {
    "learning_rate_backbone": 2e-5,
    "learning_rate_heads": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "max_epochs": 4,
    "alpha_loss": 0.5,    # Trọng số L_MSE (NSS) [Mục IV.3]
    "beta_loss": 0.3,     # Trọng số L_CE (Tier) [Mục IV.3]
    "lambda_kd_loss": 0.2 # Trọng số L_KD (Knowledge Distillation) [Mục IV.3]
}

# Ngưỡng đồng thuận Multi-Teacher Ensemble (Consensus Filtering) [Mục IV.2.2]
CONSENSUS_MAX_STD_DEV: Final[float] = 0.25
NEUTRAL_ZONE_BOUNDS: Final[tuple] = (-0.05, 0.05)