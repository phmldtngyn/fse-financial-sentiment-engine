import os
import json
import time
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GeminiTripleTeacher")

SYSTEM_PROMPT = """Bạn là một chuyên gia phân tích rủi ro tài chính và cảm xúc thị trường chứng khoán Việt Nam.
Nhiệm vụ của bạn là đánh giá bài báo tài chính được cung cấp và trả về kết quả dưới dạng JSON duy nhất.

Đầu ra YÊU CẦU định dạng JSON chuẩn với 2 trường:
1. "NSS": float từ -1.0 (Rất tiêu cực) đến 1.0 (Rất tích cực).
2. "Tier": int từ 1 đến 4 đại diện cho độ liên quan:
   - 1: Liên quan trực tiếp đến mã ngân hàng/cổ phiếu cụ thể.
   - 2: Liên quan đến ngành tài chính - ngân hàng nói chung.
   - 3: Liên quan đến kinh tế vĩ mô (lãi suất điều hành, GDP, tỷ giá, CPI).
   - 4: Tin lề đường, không liên quan.
"""

class GeminiTripleTeacher:
    def __init__(self):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key or gemini_key.startswith("AIzaSy_KEY"):
            raise ValueError("Chưa khai báo GEMINI_API_KEY chuẩn trong biến môi trường!")
        self.client = genai.Client(api_key=gemini_key)

    def _call_gemini_with_retry(self, model_name: str, text: str, max_retries: int = 3) -> dict:
        # Cấu hình buộc trả về Structured JSON Output để không bị lỗi parse
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json"
        )
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=f"Bài báo tài chính:\n{text}",
                    config=config
                )
                return json.loads(response.text.strip())
            except Exception as e:
                logger.warning(f"Lần thử {attempt + 1}/{max_retries} lỗi ({model_name}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        return None

    def call_teacher_1(self, text: str) -> dict:
        # Teacher 1: Gemini 2.5 Flash
        return self._call_gemini_with_retry('gemini-2.5-flash', text)

    def call_teacher_2(self, text: str) -> dict:
        # Teacher 2: Gemini 1.5 Flash Latest (Alias tương thích v1beta)
        res = self._call_gemini_with_retry('gemini-1.5-flash-latest', text)
        if not res:
            res = self._call_gemini_with_retry('gemini-2.5-pro', text)
        return res

    def call_teacher_3(self, text: str) -> dict:
        # Teacher 3: Gemini 1.5 Pro Latest
        return self._call_gemini_with_retry('gemini-1.5-pro-latest', text)

    def filter_consensus(self, res_t1: dict, res_t2: dict, res_t3: dict) -> dict:
        valid_nss = []
        valid_tiers = []

        for res in [res_t1, res_t2, res_t3]:
            if res and isinstance(res, dict) and "NSS" in res and "Tier" in res:
                valid_nss.append(float(res["NSS"]))
                valid_tiers.append(int(res["Tier"]))

        if len(valid_nss) < 2:
            return None

        y_consensus = float(np.mean(valid_nss))
        sigma = float(np.std(valid_nss))

        if sigma > 0.35:
            return None

        tier_consensus = int(round(np.mean(valid_tiers)))

        return {
            "y_consensus": round(y_consensus, 4),
            "tier_consensus": tier_consensus,
            "sigma": round(sigma, 4),
            "soft_labels": {
                "t1_flash_25": res_t1.get("NSS") if res_t1 else None,
                "t2_flash_15": res_t2.get("NSS") if res_t2 else None,
                "t3_pro_15": res_t3.get("NSS") if res_t3 else None
            }
        }

def run_labeling_pipeline():
    processed_dir = Path("data/processed")
    gold_dir = Path("data/gold_standard")
    gold_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(processed_dir.glob("processed_data_*.json"))
    if not json_files:
        logger.error("Không tìm thấy file processed_data_*.json nào!")
        return

    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Đang nạp bài báo từ: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    try:
        teachers = GeminiTripleTeacher()
    except Exception as e:
        logger.error(f"Khởi tạo thất bại: {e}")
        return

    labeled_dataset = []
    logger.info(f"Bắt đầu gán nhãn 3 Teachers Gemini chuẩn cho {len(articles)} bài báo...")

    for idx, article in enumerate(articles):
        text_content = f"Tiêu đề: {article.get('title', '')}\nNội dung: {article.get('body', '')[:1500]}"
        
        res_t1 = teachers.call_teacher_1(text_content)
        res_t2 = teachers.call_teacher_2(text_content)
        res_t3 = teachers.call_teacher_3(text_content)

        consensus = teachers.filter_consensus(res_t1, res_t2, res_t3)

        if consensus:
            labeled_item = article.copy()
            labeled_item.update(consensus)
            labeled_dataset.append(labeled_item)
            logger.info(f"[{idx+1}/{len(articles)}] PASSED | NSS: {consensus['y_consensus']} | Tier: {consensus['tier_consensus']} | Sigma: {consensus['sigma']}")
        else:
            logger.warning(
                f"[{idx+1}/{len(articles)}] REJECTED | "
                f"T1: {res_t1 is not None}, T2: {res_t2 is not None}, T3: {res_t3 is not None}"
            )

        time.sleep(1.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = gold_dir / f"distillation_dataset_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_dataset, f, ensure_ascii=False, indent=4)

    logger.info(f"Hoàn thành! Đã tạo tập nhãn sạch đạt chuẩn Distillation với {len(labeled_dataset)}/{len(articles)} bài báo. Lưu tại: {output_file}")

if __name__ == "__main__":
    run_labeling_pipeline()