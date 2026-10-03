import re
import unicodedata
from pyvi import ViTokenizer

class VietnameseTextNormalizer:
    def __init__(self):
        pass

    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa Unicode về dạng NFC (chuẩn gõ tiếng Việt)."""
        if not text:
            return ""
        return unicodedata.normalize('NFC', text)

    def clean_raw_text(self, text: str) -> str:
        """Loại bỏ ký tự điều khiển, html tags dư thừa, chuẩn hóa khoảng trắng."""
        if not text:
            return ""
        
        # Chuẩn hóa Unicode NFC
        text = self.normalize_unicode(text)
        
        # Loại bỏ các ký tự điều khiển / xuống dòng thừa
        text = re.sub(r'[\r\n\t]+', ' ', text)
        
        # Loại bỏ nhiều khoảng trắng liên tiếp
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def word_segment(self, text: str) -> str:
        """
        Tách từ hình thái học tiếng Việt.
        Biến các từ ghép thành dạng nối gạch dưới (VD: ngân hàng -> ngân_hàng).
        """
        cleaned_text = self.clean_raw_text(text)
        if not cleaned_text:
            return ""
        
        # Sử dụng ViTokenizer để tách từ
        segmented_text = ViTokenizer.tokenize(cleaned_text)
        return segmented_text

    def process_article(self, article: dict) -> dict:
        """Xử lý toàn bộ bài báo: chuẩn hóa cả title và body."""
        processed_article = article.copy()
        
        title_raw = article.get('title', '')
        body_raw = article.get('body', '')
        
        processed_article['title_normalized'] = self.clean_raw_text(title_raw)
        processed_article['body_normalized'] = self.clean_raw_text(body_raw)
        
        # Tách từ phục vụ PhoBERT
        processed_article['title_segmented'] = self.word_segment(title_raw)
        processed_article['body_segmented'] = self.word_segment(body_raw)
        
        return processed_article