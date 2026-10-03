try:
    from constants import RELEVANCE_TIER_WEIGHTS as TIER_WEIGHTS
except ImportError:
    try:
        from constants import TIER_WEIGHTS
    except ImportError:
        # Baseline mặc định theo Spec nếu chưa định nghĩa trong constants.py
        TIER_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.3, 4: 0.0}

class SentimentIndexAggregator:
    def __init__(self, gamma: float = 1.75, lambda_param: float = 0.60):
        self.gamma = gamma       # Prospect Theory multiplier cho tin xấu (baseline 1.75)
        self.lambda_param = lambda_param # Quán tính cảm xúc ngày hôm trước S_{t-1} (baseline 0.60)
        self.tier_weights = TIER_WEIGHTS

    def calculate_raw_sentiment(self, articles_data: list) -> float:
        """
        Tính S_t^{raw} dựa theo công thức tổng quát Spec Section VI.2.1:
        S_t^{raw} = sum(NSS_j * Tier_j * W_j * N_j * decay) / sum(Tier_j * W_j * N_j * decay)
        """
        if not articles_data:
            return 0.0

        numerator = 0.0
        denominator = 0.0

        for item in articles_data:
            nss_j = item.get('NSS_j', 0.0)
            
            # Quy đổi Tier rời rạc sang trọng số Tier_j
            tier_num = item.get('Tier_j', 4)
            tier_val = self.tier_weights.get(tier_num, 0.0)
            
            w_j = item.get('W_j', 0.32)
            n_j = item.get('Novelty_N_j', 1.0)
            decay_j = item.get('Time_Decay_Factor', 1.0)

            # Trọng số tổng hợp cho bài báo j
            combined_weight = tier_val * w_j * n_j * decay_j

            numerator += nss_j * combined_weight
            denominator += combined_weight

        if denominator == 0.0:
            return 0.0

        s_t_raw = numerator / denominator
        # Chuẩn hóa về khoảng [-1.0, 1.0]
        return round(max(-1.0, min(1.0, s_t_raw)), 4)

    def apply_prospect_theory(self, s_t_raw: float) -> float:
        """Phi(S_t^{raw}): Phóng đại tác động của tin xấu qua hệ số gamma (Spec Section VI.3.1)"""
        if s_t_raw >= 0:
            return s_t_raw
        else:
            phi_val = self.gamma * s_t_raw
            return round(max(-1.0, phi_val), 4)

    def calculate_final_index(self, s_t_raw: float, s_previous: float = 0.0) -> float:
        """
        Tính S_t cuốn chiếu trễ tâm lý:
        S_t = lambda * S_{t-1} + (1 - lambda) * Phi(S_t^{raw}) (Spec Section VI.3.2)
        """
        phi_s_raw = self.apply_prospect_theory(s_t_raw)
        s_t = self.lambda_param * s_previous + (1 - self.lambda_param) * phi_s_raw
        return round(max(-1.0, min(1.0, s_t)), 4)