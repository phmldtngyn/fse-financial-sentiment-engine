"""
Model mini CHỈ dùng cho smoke test — thay PhoBERT để chạy end-to-end trên CPU
trong vài giây, không cần tải weights 500MB.

Factory path: tests.smoke_model:build_model (đặt trong config.model.factory).
Model trả tensor trần (không wrapper .logits) — chủ ý để kiểm tra luôn nhánh
xử lý output trong evaluate.predict().
"""

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 16, num_labels: int = 3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.head = nn.Linear(dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        emb = self.emb(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def build_model(model_cfg: dict) -> TinyModel:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    return TinyModel(vocab_size=len(tok), num_labels=int(model_cfg.get("num_labels", 3)))
