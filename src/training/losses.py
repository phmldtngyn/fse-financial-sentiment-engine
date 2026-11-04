"""
Loss cho phân loại sentiment 3 lớp (FSE).

Sửa so với bản cũ:
- Class weights KHÔNG còn hardcode [2.75, 0.50, 1.54] trong code. Mỗi lần
  retrain, trọng số được tính ĐỘNG từ tập train theo nghịch đảo tần suất
  (w_c = N / (K * n_c)) — đúng công thức sinh ra bộ số cũ, nhưng tự thích ứng
  khi phân bố dữ liệu thay đổi theo thời gian.
- Hỗ trợ label smoothing (giảm overconfidence — có ý nghĩa khi S_t được tính
  từ nhãn argmax cứng) và focal loss (giúp lớp negative chiếm ~12%).

Lưu ý: label_smoothing và focal_loss loại trừ lẫn nhau (config đã ghi chú).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["negative", "neutral", "positive"]


def compute_class_weights(labels, num_labels: int = 3) -> np.ndarray:
    """Trọng số nghịch đảo tần suất: w_c = N / (K * n_c), chuẩn hóa mean(w) = 1.

    Ví dụ với phân bố 12.1% / 66.3% / 21.6% (n=5166) cho đúng bộ
    [2.75, 0.50, 1.54] mà bản cũ hardcode.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_labels).astype(float)
    counts = np.clip(counts, 1.0, None)  # chống chia 0 khi một lớp vắng mặt hoàn toàn
    weights = counts.sum() / (num_labels * counts)
    return weights


class FSELoss3Class(nn.Module):
    """Weighted CrossEntropy / Focal loss cho 3 lớp sentiment.

    Tham số:
        class_weights: "auto" (tính từ train_labels) hoặc list/tensor 3 số.
        train_labels:  nhãn tập train (bắt buộc khi class_weights="auto").
        label_smoothing: 0.0–1.0, chỉ dùng khi focal=False.
        focal: bật focal loss thay cho cross-entropy.
        focal_gamma: gamma của focal loss.
    """

    def __init__(
        self,
        num_labels: int = 3,
        class_weights="auto",
        train_labels=None,
        label_smoothing: float = 0.0,
        focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        if focal and label_smoothing > 0:
            raise ValueError("label_smoothing và focal_loss loại trừ lẫn nhau — chỉ chọn một.")

        if isinstance(class_weights, str) and class_weights == "auto":
            if train_labels is None:
                raise ValueError('class_weights="auto" cần truyền train_labels (nhãn tập train).')
            weights = torch.tensor(
                compute_class_weights(train_labels, num_labels), dtype=torch.float32
            )
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.numel() != num_labels:
                raise ValueError(f"class_weights phải có {num_labels} phần tử.")
        self.register_buffer("weights", weights)

        self.num_labels = num_labels
        self.label_smoothing = float(label_smoothing)
        self.focal = bool(focal)
        self.focal_gamma = float(focal_gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """Trả về (loss, components) — giữ tương thích signature bản cũ."""
        labels = labels.long()

        if self.focal:
            # Focal loss có trọng số lớp: FL = w_y * (1 - p_y)^gamma * (-log p_y)
            logp = F.log_softmax(logits.float(), dim=-1)
            logp_y = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
            p_y = logp_y.exp()
            w_y = self.weights[labels]
            loss = -(w_y * (1.0 - p_y) ** self.focal_gamma * logp_y).mean()
            components = {"focal": loss.detach()}
        else:
            loss = F.cross_entropy(
                logits.float(),
                labels,
                weight=self.weights,
                label_smoothing=self.label_smoothing,
            )
            components = {"ce": loss.detach()}

        return loss, components
