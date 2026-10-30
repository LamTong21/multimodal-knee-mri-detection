import torch
import torch.nn as nn


class MaskedAsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) kết hợp Masking nhãn chưa gán (-1 hoặc NaN):
    - Đè nén gradient từ negative samples dễ dàng (gamma_neg > gamma_pos)
    - Áp dụng probability clipping margin cho negative
    - Bỏ qua loss tại các nhãn không xác định
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = (targets >= 0.0) & (~torch.isnan(targets))
        targets_clean = torch.where(valid_mask, targets, torch.zeros_like(targets))

        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets_clean * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets_clean) * torch.log(probs_neg.clamp(min=self.eps))
        loss = -(loss_pos + loss_neg)

        weight = targets_clean * torch.pow(1.0 - probs_pos, self.gamma_pos) + \
                 (1.0 - targets_clean) * torch.pow(1.0 - probs_neg, self.gamma_neg)
        loss = loss * weight

        masked_loss = loss * valid_mask.float()
        return masked_loss.sum() / valid_mask.float().sum().clamp(min=1.0)