"""Loss cho 3-class sentiment classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSELoss3Class(nn.Module):
    """CrossEntropy với class weights để xử lý imbalance.

    Default weights từ train distribution (12% neg / 66% neu / 22% pos):
    - negative: 2.75 (hiếm → weight cao)
    - neutral:  0.50 (nhiều → weight thấp)
    - positive: 1.54
    """

    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            class_weights = [2.75, 0.50, 1.54]
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float),
        )

    def forward(self, logits, labels):
        """logits: (B, 3), labels: (B,) long. Returns (loss, dict)."""
        loss = F.cross_entropy(
            logits, labels, weight=self.class_weights.to(logits.device)
        )
        return loss, {"ce": loss.item()}