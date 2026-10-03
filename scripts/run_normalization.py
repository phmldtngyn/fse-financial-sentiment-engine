import json
import logging
from pathlib import Path
from datetime import datetime
from src.utils.vietnamese_nlp import VietnameseTextNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NormalizationPipeline")

def run_normalization():
    parsed_dir = Path("data/parsed")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tìm file JSON mới nhất trong data/parsed/
    json_files = list(parsed_dir.glob("crawled_data_*.json"))
    if not json_files:
        logger.error("Không tìm thấy file parsed JSON nào trong data/parsed/")
        return

    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Đang đọc dữ liệu từ: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # 2. Khởi tạo Normalizer
    normalizer = VietnameseTextNormalizer()
    processed_articles = []

    # 3. Tiến hành chuẩn hóa từng bài
    for item in articles:
        processed_item = normalizer.process_article(item)
        processed_articles.append(processed_item)

    # 4. Lưu ra thư mục data/processed/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = processed_dir / f"processed_data_{timestamp}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=4)

    logger.info(f"Hoàn thành Normalize {len(processed_articles)} bài báo. Đã lưu tại: {output_path}")

if __name__ == "__main__":
    run_normalization()