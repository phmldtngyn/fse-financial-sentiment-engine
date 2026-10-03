import faiss
import numpy as np
import math
from datetime import datetime, timedelta

class FAISSVectorStore:
    def __init__(self, dim=768, kappa=2.0):
        self.dim = dim
        self.kappa = kappa
        
        # Sử dụng IndexFlatIP (Inner Product) cho Cosine Similarity 
        # (yêu cầu vector đầu vào phải được chuẩn hóa L2)
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (vector / norm).astype('float32')

    def calculate_novelty_score(self, max_sim: float) -> float:
        """Tính hệ số làm mới phi tuyến (Spec Section V.3)"""
        if max_sim >= 0.85:
            return 0.0
        elif max_sim < 0.3:
            return 1.0
        else:
            return math.exp(-self.kappa * ((max_sim - 0.3) / 0.55))

    def add_and_evaluate(self, article_id: str, publish_time: str, vector: np.ndarray) -> tuple:
        """
        Quy trình: 
        1. Tìm kiếm bài báo trùng lặp trong 48h quá khứ.
        2. Tính Novelty Score.
        3. Lưu bài báo mới vào CSDL Vector.
        """
        v_norm = self._normalize_vector(vector).reshape(1, -1)
        
        novelty_score = 1.0
        max_sim = 0.0
        
        # Truy vấn tương đồng nếu index đã có dữ liệu
        if self.index.ntotal > 0:
            # Lấy top 5 bài báo gần nhất
            k = min(5, self.index.ntotal)
            similarities, indices = self.index.search(v_norm, k)
            
            # Lọc theo cửa sổ thời gian 48 giờ quá khứ
            current_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            past_48h_limit = current_time - timedelta(hours=48)
            
            valid_similarities = []
            for sim, idx in zip(similarities[0], indices[0]):
                past_article_time_str = self.metadata[idx]['publish_time']
                past_article_time = datetime.fromisoformat(past_article_time_str.replace('Z', '+00:00'))
                
                if past_48h_limit <= past_article_time <= current_time:
                    valid_similarities.append(sim)
                    
            if valid_similarities:
                max_sim = max(valid_similarities)
                novelty_score = self.calculate_novelty_score(max_sim)

        # Lưu Vector và Metadata mới vào Index
        self.index.add(v_norm)
        self.metadata.append({
            "article_id": article_id,
            "publish_time": publish_time,
            "novelty_score": novelty_score,
            "max_sim": float(max_sim)
        })
        
        return novelty_score, max_sim