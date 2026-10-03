import json
import logging
from pathlib import Path
from datetime import datetime

from src.index_engine.weights import ArticleWeightCalculator
from src.index_engine.intraday_decay import IntradayDecayCalculator
from src.index_engine.aggregator import SentimentIndexAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketIndexEngine")

def run_index_calculation():
    processed_dir = Path("data/processed")
    
    # 1. Nạp file inferred JSON mới nhất
    inferred_files = list(processed_dir.glob("inferred_data_*.json"))
    if not inferred_files:
        logger.error("Không tìm thấy file inferred_data nào trong data/processed/")
        return

    latest_file = max(inferred_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Đang nạp dữ liệu suy luận từ: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # 2. Khởi tạo các cỗ máy tính toán
    weight_calc = ArticleWeightCalculator()
    decay_calc = IntradayDecayCalculator(mu=0.05)
    aggregator = SentimentIndexAggregator(gamma=1.75, lambda_param=0.60)

    calculation_time = datetime.now().isoformat()
    processed_articles = []

    # 3. Tính toán trọng số thành phần từng bài báo
    for item in articles:
        p_tier = item.get('position_tier', 'P4')
        source = item.get('source', 'cafef')
        pub_time = item.get('publish_time', calculation_time)

        # Trọng số W_j
        w_j = weight_calc.calculate_article_weight(p_tier, source)
        
        # Suy giảm Intraday Decay
        decay_factor, delta_t = decay_calc.calculate_decay_factor(pub_time, calculation_time)

        item_copy = item.copy()
        item_copy.update({
            "W_j": w_j,
            "Time_Decay_Factor": decay_factor,
            "Trading_Hours_Delta": delta_t
        })
        processed_articles.append(item_copy)

    # 4. Tính toán Chỉ số Cảm xúc Thị trường
    s_t_raw = aggregator.calculate_raw_sentiment(processed_articles)
    phi_s_t = aggregator.apply_prospect_theory(s_t_raw)
    
    # Mặc định S_{t-1} = 0.0 cho phiên khởi tạo đầu tiên
    s_previous = 0.0 
    s_t_final = aggregator.calculate_final_index(s_t_raw, s_previous)

    logger.info("=== KẾT QUẢ CHỈ SỐ CẢM XÚC THỊ TRƯỜNG S_t ===")
    logger.info(f"Tổng số bài báo xử lý (N_t): {len(processed_articles)}")
    logger.info(f"Chỉ số Cảm xúc Thô (S_t^{{raw}}): {s_t_raw}")
    logger.info(f"Sau Prospect Theory Phi(S_t^{{raw}}): {phi_s_t}")
    logger.info(f"CHỈ SỐ CẢM XÚC DÒNG TIỀN CHÍNH THỨC (S_t): {s_t_final}")

    # 5. Lưu báo cáo kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = processed_dir / f"final_market_index_{timestamp}.json"
    
    report_data = {
        "calculation_time": calculation_time,
        "total_articles": len(processed_articles),
        "s_t_raw": s_t_raw,
        "phi_s_t_raw": phi_s_t,
        "s_t_final": s_t_final,
        "parameters": {"gamma": 1.75, "lambda": 0.60, "mu": 0.05},
        "articles": processed_articles
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Đã lưu báo cáo Chỉ số Cảm xúc Thị trường tại: {output_path}")

if __name__ == "__main__":
    run_index_calculation()