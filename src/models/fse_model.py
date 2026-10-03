import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class AttentionPooling(nn.Module):
    """
    Tầng 2: Nén các đoạn văn bản (chunks) thành 1 Vector Đại diện Bài báo duy nhất
    thông qua cơ chế Attention Weighting tự động gán trọng số (Spec Section III.2).
    """
    def __init__(self, hidden_size=768, attention_size=768):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, attention_size)
        self.u_a = nn.Linear(attention_size, 1, bias=False)

    def forward(self, hidden_states):
        # hidden_states: [num_chunks, hidden_size]
        e_k = torch.tanh(self.W_a(hidden_states))
        scores = self.u_a(e_k).squeeze(-1)  # [num_chunks]
        alpha_k = torch.softmax(scores, dim=0)
        
        # V_article = sum(alpha_k * h_k)
        v_article = torch.sum(alpha_k.unsqueeze(-1) * hidden_states, dim=0)
        return v_article


class FSEModel(nn.Module):
    def __init__(self, model_name="VinAI/phobert-base-v2"):
        super().__init__()
        # Tầng 1: Backbone Encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Tầng 2: Attention Pooling
        self.attention_pooling = AttentionPooling(hidden_size=768)
        
        # Tầng 3: Multi-Task Heads
        # 3.1 Sentiment Regression Head (NSS_j in [-1, 1])
        self.sentiment_head = nn.Linear(768, 1)
        
        # 3.2 Relevance Tier Head (CORAL Ordinal Regression cho 4 Tiers)
        self.tier_head = nn.Linear(768, 3)

    @torch.no_grad()
    def forward(self, segmented_text: str):
        """
        Quy trình trích xuất Vector, Điểm cảm xúc (NSS) và Phân tầng (Tier).
        Tự động phân đoạn (Chunking) nếu văn bản vượt quá 256 tokens.
        """
        # Phân đoạn văn bản (Chunking) để vượt giới hạn 512 tokens
        tokens = self.tokenizer(segmented_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
        max_length = 250
        chunks = []
        
        for i in range(0, tokens.size(0), max_length):
            chunk = tokens[i:i + max_length]
            if len(chunk) < 5: 
                continue
            
            # Gắn token CLS và SEP cho từng chunk
            chunk_with_special = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]), 
                chunk, 
                torch.tensor([self.tokenizer.sep_token_id])
            ])
            chunks.append(chunk_with_special)
            
        if not chunks:
            return None, 0.0, 4

        # Chạy qua Backbone
        chunk_embeddings = []
        for chunk in chunks:
            inputs = chunk.unsqueeze(0)
            outputs = self.backbone(inputs)
            # Lấy vector [CLS] của đoạn
            cls_embedding = outputs.last_hidden_state[0, 0, :]
            chunk_embeddings.append(cls_embedding)
                
        chunk_embeddings = torch.stack(chunk_embeddings)  # [num_chunks, 768]
        
        # Nén qua Attention Pooling
        v_article = self.attention_pooling(chunk_embeddings)  # [768]
        
        # Đưa qua Multi-Task Heads
        # Điểm cảm xúc ròng [-1, 1]
        nss_score = torch.tanh(self.sentiment_head(v_article)).item()
        
        # Dự đoán Tier theo phân phối Ordinal (1 -> 4)
        tier_logits = torch.sigmoid(self.tier_head(v_article))
        tier_pred = 1 + torch.sum(tier_logits > 0.5).item()
        
        # Chuyển đổi an toàn về NumPy bằng detach().cpu().numpy()
        return v_article.detach().cpu().numpy(), nss_score, tier_pred