import math
from datetime import datetime, time, timedelta

class IntradayDecayCalculator:
    def __init__(self, mu: float = 0.05):
        self.mu = mu # Tốc độ suy giảm thời gian thực (baseline = 0.05)

    def is_trading_day(self, dt: datetime) -> bool:
        """Kiểm tra ngày giao dịch (Loại trừ Thứ 7, Chủ Nhật)"""
        return dt.weekday() < 5

    def calculate_trading_hours_diff(self, start_dt: datetime, end_dt: datetime) -> float:
        """
        Tính tổng số giờ giao dịch thực tế giữa publish_time và thời điểm tính S_t.
        Bỏ qua thời gian nghỉ trưa, qua đêm và cuối tuần (Spec Section VI.2.2).
        """
        if start_dt >= end_dt:
            return 0.0

        trading_hours = 0.0
        current = start_dt

        # Duyệt qua từng khung thời gian
        while current < end_dt:
            if self.is_trading_day(current):
                t = current.time()
                # Khung sáng: 09:00 - 11:30 (2.5 giờ)
                if time(9, 0) <= t < time(11, 30):
                    trading_hours += 1 / 60.0
                # Khung chiều: 13:00 - 15:00 (2.0 giờ)
                elif time(13, 0) <= t < time(15, 0):
                    trading_hours += 1 / 60.0
            
            current += timedelta(minutes=1)

        return trading_hours / 60.0 # Quy đổi về giờ giao dịch

    def calculate_decay_factor(self, publish_time_str: str, calculation_time_str: str) -> tuple:
        """Tính exp(-mu * delta_t_j)"""
        try:
            pub_dt = datetime.fromisoformat(publish_time_str.replace('Z', '+00:00'))
            calc_dt = datetime.fromisoformat(calculation_time_str.replace('Z', '+00:00'))
        except ValueError:
            pub_dt = datetime.strptime(publish_time_str[:10], "%Y-%m-%d")
            calc_dt = datetime.strptime(calculation_time_str[:10], "%Y-%m-%d")

        delta_t_j = self.calculate_trading_hours_diff(pub_dt, calc_dt)
        decay_factor = math.exp(-self.mu * delta_t_j)
        
        return round(decay_factor, 4), round(delta_t_j, 2)