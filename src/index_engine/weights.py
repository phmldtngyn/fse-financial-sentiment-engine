import logging

logger = logging.getLogger("ArticleWeightCalculator")

# Import an toàn từ constants.py, có fallback baseline chuẩn theo Spec (Section VI.1.2 & VIII.2)
try:
    from constants import POSITION_TIER_WEIGHTS
except ImportError:
    POSITION_TIER_WEIGHTS = {
        "P1": 1.0, "P2": 0.8, "P3": 0.6, "P4": 0.4, "P5": 0.2, "P6": 0.1
    }

try:
    from constants import SOURCE_CREDIBILITY_BASELINE
except ImportError:
    try:
        from constants import SOURCE_CREDIBILITY as SOURCE_CREDIBILITY_BASELINE
    except ImportError:
        # Fallback baseline chuẩn cho 04 nguồn theo Spec
        SOURCE_CREDIBILITY_BASELINE = {
            "thoibaotaichinhvietnam": 1.0,
            "vietnamfinance": 1.0,
            "cafef": 0.8,
            "vietstock": 0.8
        }


class ArticleWeightCalculator:
    def __init__(self, custom_source_credibility: dict = None):
        self.position_weights = POSITION_TIER_WEIGHTS
        self.source_credibility = custom_source_credibility or SOURCE_CREDIBILITY_BASELINE

    def get_position_weight(self, position_tier: str) -> float:
        """Lấy PositionWeight theo phân tầng P1-P6 (Spec Section VI.1.1)"""
        return self.position_weights.get(position_tier, 0.4) # Mặc định P4 = 0.4

    def get_source_credibility(self, source: str) -> float:
        """Lấy SourceCredibility cho 04 nguồn tin (Spec Section VI.1.2)"""
        source_key = source.lower().strip()
        return self.source_credibility.get(source_key, 0.8) # Baseline mặc định 0.8

    def calculate_article_weight(self, position_tier: str, source: str) -> float:
        """W_j = PositionWeight_j * SourceCredibility_j (Spec Section VI.1)"""
        p_weight = self.get_position_weight(position_tier)
        s_cred = self.get_source_credibility(source)
        return round(p_weight * s_cred, 4)