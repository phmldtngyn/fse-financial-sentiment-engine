import torch.nn as nn
import torch.nn.functional as F


class FSELoss(nn.Module):
    """Combined loss cho SFT: λ₁·MSE(score) + λ₂·CrossEntropy(tier).

    - MSE: hồi quy score cảm xúc liên tục ∈ [-1, 1]
    - CE:  phân loại tier ∈ {0, 1, 2} (0-indexed từ dataset)
    """

    def __init__(self, lambda_mse=1.0, lambda_ce=0.5):
        super().__init__()
        self.lm = lambda_mse
        self.lc = lambda_ce
        self.mse = nn.MSELoss()

    def forward(self, score_pred, tier_logits, score_true, tier_true):
        l_mse = self.mse(score_pred, score_true)
        l_ce = F.cross_entropy(tier_logits, tier_true)
        total = self.lm * l_mse + self.lc * l_ce
        return total, {"mse": l_mse.item(), "ce": l_ce.item()}