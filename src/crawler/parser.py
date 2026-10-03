import trafilatura
from bs4 import BeautifulSoup
from datetime import datetime

class ArticleParser:
    def parse(self, html_content: str, item_meta: dict) -> dict:
        """Trích xuất 10 trường dữ liệu bắt buộc và mở rộng theo chuẩn Spec Section II.2."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. Trích xuất body sạch (loại bỏ menu, quảng cáo, bài liên quan)
        extracted_text = trafilatura.extract(
            html_content, 
            include_comments=False, 
            include_tables=False, 
            no_fallback=False
        )
        
        if not extracted_text:
            return {}

        # 2. Trích xuất Metadata
        metadata = trafilatura.extract_metadata(html_content)
        
        # Title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ""
        if not title and metadata and metadata.title:
            title = metadata.title

        # Publish Time (định dạng ISO 8601)
        publish_time = metadata.date if (metadata and metadata.date) else datetime.now().isoformat()

        # 3. Phân tầng Vị trí (P1 - P6) dựa theo đặc tả Section VI.1.1
        position_tier = self._determine_position_tier(soup, item_meta)

        return {
            "title": title,
            "body": extracted_text,
            "publish_time": publish_time,
            "author": metadata.author if (metadata and metadata.author) else "",
            "category": item_meta.get('category', ''),
            "url": item_meta['url'],
            "source": item_meta['source'],
            "position_tier": position_tier,
            "position_source": item_meta.get('position_source', 'category'),
            "position_captured_at": datetime.now().isoformat()
        }

    def _determine_position_tier(self, soup: BeautifulSoup, item_meta: dict) -> str:
        """
        Gán nhãn tự động P1-P6 theo quy định tại Section VI.1.1:
        - P1: Bài nổi bật trang chủ (Hero/Featured)
        - P2: Bài trang chủ nằm trong khối chính
        - P3: Bài nổi bật trang chuyên mục Ngân hàng/Tài chính
        - P4: Bài thuộc danh sách chuyên mục (Default)
        - P5: Bài chỉ xuất hiện trang chi tiết / tìm kiếm
        - P6: Bài trong khối tin liên quan / cũ đào lại
        """
        try:
            # 1. Kiểm tra P6: Tin nằm trong khối Bài viết liên quan
            if soup.find('aside') or soup.find(class_=lambda c: c and any(x in c.lower() for x in ['related', 'tin-lien-quan', 'box-news-related'])):
                return "P6"

            # 2. Kiểm tra P1: Bài Hero / Featured
            hero_selectors = ['hero', 'featured', 'top-story', 'focus-news', 'highlight']
            if soup.find(class_=lambda c: c and any(x in c.lower() for x in hero_selectors)):
                return "P1"

            # 3. Kiểm tra P2: Trang chủ khối nội dung chính
            if item_meta.get('position_source') == 'homepage':
                return "P2"

            # 4. Kiểm tra P3: Bài nổi bật chuyên mục
            category_top = ['cate-highlight', 'top-cate', 'first-news']
            if soup.find(class_=lambda c: c and any(x in c.lower() for x in category_top)):
                return "P3"

            # 5. Trường hợp không xác định selector mặc định gán P4 theo đúng Spec
            return "P4"
        except Exception:
            return "P4"