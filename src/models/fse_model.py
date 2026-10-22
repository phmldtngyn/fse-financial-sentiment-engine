import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class AttentionPooling(nn.Module):
    """Nén các chunk embeddings → 1 vector bài báo qua attention weighting."""

    def __init__(self, hidden_size=768, attention_size=768):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, attention_size)
        self.u_a = nn.Linear(attention_size, 1, bias=False)

    def forward(self, hidden_states):
        # hidden_states: (num_chunks, hidden_size)
        e_k = torch.tanh(self.W_a(hidden_states))
        scores = self.u_a(e_k).squeeze(-1)  # (num_chunks,)
        alpha_k = torch.softmax(scores, dim=0)
        v_article = torch.sum(alpha_k.unsqueeze(-1) * hidden_states, dim=0)
        return v_article


class FSEModel(nn.Module):
    """FSE: PhoBERT backbone + AttentionPooling + 3-class sentiment head.

    Class labels: 0 = negative, 1 = neutral, 2 = positive
    """

    def __init__(self, model_name="VinAI/phobert-base-v2", num_classes=3):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.attention_pooling = AttentionPooling(hidden_size=768)

        # 3-class classification head
        self.sentiment_head = nn.Linear(768, num_classes)

    @torch.no_grad()
    def forward(self, segmented_text: str):
        """Inference cho bài báo dài (chunking nếu > 250 tokens).

        Returns:
            (v_article_np, sentiment_logits_np) — v_article: (768,), logits: (3,)
            Hoặc (None, None) nếu text rỗng.
        """
        device = next(self.parameters()).device

        tokens = self.tokenizer(
            segmented_text, return_tensors="pt", truncation=False
        )["input_ids"].squeeze(0)

        max_length = 250
        chunks = []
        for i in range(0, tokens.size(0), max_length):
            chunk = tokens[i:i + max_length]
            if len(chunk) < 5:
                continue
            chunk_with_special = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),
                chunk,
                torch.tensor([self.tokenizer.sep_token_id]),
            ]).to(device)
            chunks.append(chunk_with_special)

        if not chunks:
            return None, None

        chunk_embeddings = []
        for chunk in chunks:
            outputs = self.backbone(input_ids=chunk.unsqueeze(0))
            chunk_embeddings.append(outputs.last_hidden_state[0, 0, :])

        chunk_embeddings = torch.stack(chunk_embeddings)  # (num_chunks, 768)
        v_article = self.attention_pooling(chunk_embeddings)  # (768,)

        sentiment_logits = self.sentiment_head(v_article)  # (3,)

        return (
            v_article.cpu().numpy(),
            sentiment_logits.cpu().numpy(),
        )

    def forward_batch(self, input_ids, attention_mask):
        """Batch forward cho SFT training (3-class classification).

        Args:
            input_ids: (B, L)
            attention_mask: (B, L)

        Returns:
            sentiment_logits: (B, 3)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        sentiment_logits = self.sentiment_head(cls)  # (B, 3)
        return sentiment_logits