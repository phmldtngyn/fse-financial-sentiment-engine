import json
import torch
import logging
from pathlib import Path
from datetime import datetime

from src.models.fse_model import FSEModel
from src.storage.vector_db import FAISSVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FSEInferencePipeline")

def run_inference():
    processed_dir = Path("data/processed")
    
    # 1. Tìm file processed JSON mới nhất
    json_files = list(processed_dir.glob("processed_data_*.json"))
    if not json_files:
        logger.error("Không tìm thấy file processed JSON nào trong data/processed/")
        return

    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Đang nạp dữ liệu từ: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # 2. Khởi tạo mô hình FSE và FAISS Vector Store
    logger.info("Đang khởi tạo mô hình FSE PhoBERT-base...")
    fse_model = FSEModel()
    fse_model.eval() # Chuyển sang chế độ suy luận
    
    vector_store = FAISSVectorStore(dim=768, kappa=2.0)
    
    inferred_articles = []

    # 3. Tiến hành suy luận và tạo Vector Store
    logger.info(f"Đang suy luận FSE & FAISS cho {len(articles)} bài báo...")
    for idx, item in enumerate(articles):
        segmented_text = item.get('body_segmented', '')
        if not segmented_text:
            continue
            
        # Suy luận FSE
        v_article, nss_score, tier_pred = fse_model(segmented_text)
        
        if v_article is None:
            continue

        # Tính Novelty Score và lưu Vector vào FAISS DB
        article_id = item.get('url', f"article_{idx}")
        publish_time = item.get('publish_time', datetime.now().isoformat())
        
        novelty_score, max_sim = vector_store.add_and_evaluate(
            article_id=article_id,
            publish_time=publish_time,
            vector=v_article
        )

        # Ghi nhận kết quả suy luận
        inferred_item = item.copy()
        inferred_item.update({
            "NSS_j": round(nss_score, 4),
            "Tier_j": tier_pred,
            "Novelty_N_j": round(novelty_score, 4),
            "Max_Similarity": round(float(max_sim), 4)
        })
        
        inferred_articles.append(inferred_item)
        logger.info(f"[{idx+1}/{len(articles)}] Article: {item['title'][:40]}... | NSS: {nss_score:.3f} | Tier: {tier_pred} | Novelty N_j: {novelty_score:.3f}")

    # 4. Lưu kết quả suy luận
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = processed_dir / f"inferred_data_{timestamp}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inferred_articles, f, ensure_ascii=False, indent=4)

    logger.info(f"Hoàn thành Suy luận FSE. Đã lưu dữ liệu tại: {output_path}")

if __name__ == "__main__":
    run_inference()