import hashlib
from datetime import datetime

class Deduplicator:
    def __init__(self):
        # Trong thực tế, nên dùng Redis hoặc DB thay vì set in-memory
        self.seen_urls = set()
        self.seen_contents = set()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def is_duplicate_url(self, url: str) -> bool:
        """Kiểm tra URL-level deduplication"""
        url_hash = self._hash(url.strip().lower())
        if url_hash in self.seen_urls:
            return True
        self.seen_urls.add(url_hash)
        return False

    def is_duplicate_content(self, title: str, publish_time: str) -> bool:
        """Kiểm tra Content-level deduplication (làm mờ đến ngày)"""
        if not title or not publish_time:
            return False
            
        try:
            # Làm mờ thời gian xuống cấp độ 'Ngày'
            dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = publish_time[:10]
            
        content_key = f"{title.strip().lower()}_{date_str}"
        content_hash = self._hash(content_key)
        
        if content_hash in self.seen_contents:
            return True
        self.seen_contents.add(content_hash)
        return False