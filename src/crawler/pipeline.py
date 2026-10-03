import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from .discovery import DiscoveryEngine
from .dedup import Deduplicator
from .fetcher import AsyncFetcher
from .parser import ArticleParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CrawlerPipeline")

async def run_pipeline(sources: list):
    """Thực thi pipeline thu thập dữ liệu qua 6 bước kiến trúc."""
    logger.info(f"Khởi động Crawler Pipeline cho các nguồn: {sources}")
    
    # Khởi tạo các module
    discovery_engine = DiscoveryEngine(sources)
    dedup = Deduplicator()
    fetcher = AsyncFetcher()
    parser = ArticleParser()
    
    # Bước 1: Discovery
    discovered_items = await discovery_engine.discover_urls()
    logger.info(f"Đã phát hiện {len(discovered_items)} URLs từ các nguồn.")

    valid_articles = []
    
    # Vòng lặp xử lý từng URL theo cấu trúc tuần tự
    for item in discovered_items:
        # Bước 4: Dedup (URL-level) - Đưa lên trước để giảm tải Fetch
        if dedup.is_duplicate_url(item['url']):
            logger.debug(f"Bỏ qua URL trùng lặp: {item['url']}")
            continue
            
        # Bước 2: Fetch
        html_content = await fetcher.fetch_html(item['url'])
        if not html_content:
            continue
            
        # Bước 3: Parse & Normalize
        parsed_data = parser.parse(html_content, item)
        if not parsed_data:
            continue
            
        # Bước 4: Dedup (Content-level)
        if dedup.is_duplicate_content(parsed_data['title'], parsed_data['publish_time']):
            logger.debug(f"Bỏ qua Content trùng lặp: {parsed_data['title']}")
            continue
            
        valid_articles.append(parsed_data)
        logger.info(f"Parse thành công: {parsed_data['title']} (Tier: {parsed_data['position_tier']})")

    # Bước 5: Store (Lưu trữ tạm ra file JSON parsed)
    save_dir = Path("data/parsed")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"crawled_data_{timestamp}.json"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(valid_articles, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Hoàn thành Pipeline. Đã lưu {len(valid_articles)} bài báo tại {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSE Crawler Pipeline")
    parser.add_argument(
        "--sources", 
        type=str, 
        required=True, 
        help="Danh sách các nguồn, cách nhau bằng dấu phẩy (vd: cafef,vietstock,vneconomy,vietnamfinance)"
    )
    args = parser.parse_args()
    source_list = [s.strip() for s in args.sources.split(',')]
    
    asyncio.run(run_pipeline(source_list))