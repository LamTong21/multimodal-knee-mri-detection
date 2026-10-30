import os
import torch
import torch.nn as nn
from .backbones import MRIAttentionClassifier


def load_backbone_weights(model: nn.Module, checkpoint_path: str, device: str = "cpu"):
    """Tải trọng số pre-trained cho backbone từ checkpoint Phase 1[cite: 2]."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"--> [Fusion] Nạp trọng số backbone thành công: {checkpoint_path}")


class MRIStudyFusionClassifier(nn.Module):
    """
    Mô hình hợp nhất 3 mặt phẳng (Sagittal, Coronal, Axial) qua Cross-Plane Attention:
    Tách biệt 2 Classification Heads:
    - Nhánh 1: Dây chằng và Sụn chêm (ACL, MCL, Medial Meniscus, Lateral Meniscus)
    - Nhánh 2: Thoái hóa khớp, Tràn dịch và Chấn thương xương (8 nhãn còn lại)
    """
    def __init__(self, num_classes: int = 12, pretrained_backbones: dict = None, freeze_backbones: bool = True):
        super().__init__()
        self.sag_backbone = MRIAttentionClassifier(num_classes=num_classes, pretrained=True)
        self.cor_backbone = MRIAttentionClassifier(num_classes=num_classes, pretrained=True)
        self.axi_backbone = MRIAttentionClassifier(num_classes=num_classes, pretrained=True)

        if pretrained_backbones:
            if 'sag' in pretrained_backbones:
                load_backbone_weights(self.sag_backbone, pretrained_backbones['sag'])
            if 'cor' in pretrained_backbones:
                load_backbone_weights(self.cor_backbone, pretrained_backbones['cor'])
            if 'axi' in pretrained_backbones:
                load_backbone_weights(self.axi_backbone, pretrained_backbones['axi'])

        self.set_backbone_freeze(freeze_backbones)

        f_dim = self.sag_backbone.feature_dim  # 768
        self.plane_norm1 = nn.LayerNorm(f_dim)
        self.plane_attn = nn.MultiheadAttention(embed_dim=f_dim, num_heads=4, batch_first=True, dropout=0.3)
        self.plane_norm2 = nn.LayerNorm(f_dim)
        self.plane_ffn = nn.Sequential(
            nn.Linear(f_dim, f_dim * 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(f_dim * 2, f_dim)
        )

        fusion_dim = f_dim * 3

        # Nhánh Ligament & Meniscus (4 nhãn)
        self.ligament_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.55),
            nn.Linear(256, 4)
        )

        # Nhánh Bone & Joint Degeneration (8 nhãn)
        self.joint_bone_head = nn.Sequential(
            nn.Linear(fusion_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.55),
            nn.Linear(384, 8)
        )

    def set_backbone_freeze(self, freeze: bool = True):
        for bb in [self.sag_backbone, self.cor_backbone, self.axi_backbone]:
            for param in bb.parameters():
                param.requires_grad = not freeze

    def forward(self, sag_img: torch.Tensor, cor_img: torch.Tensor, axi_img: torch.Tensor) -> torch.Tensor:
        v_sag = self.sag_backbone.extract_features(sag_img)
        v_cor = self.cor_backbone.extract_features(cor_img)
        v_axi = self.axi_backbone.extract_features(axi_img)

        # Token tương tác liên mặt phẳng [Batch, 3, 768]
        tokens = torch.stack([v_sag, v_cor, v_axi], dim=1)
        norm_tokens = self.plane_norm1(tokens)
        attn_out, _ = self.plane_attn(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out
        tokens = tokens + self.plane_ffn(self.plane_norm2(tokens))

        fused = tokens.flatten(1)
        logits_ligament = self.ligament_head(fused)
        logits_joint = self.joint_bone_head(fused)

        return torch.cat([logits_ligament, logits_joint], dim=1)