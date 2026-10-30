import torch
import torch.nn as nn
import torchvision.models as models


class SliceTransformerAggregation(nn.Module):
    """Gom cụm đặc trưng theo trục chuỗi lát cắt MRI bằng Transformer Encoder + Attention Pooling[cite: 2]."""
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Num_Slices, Feature_Dim]
        feat = self.transformer(x)
        weights = torch.softmax(self.attn_pool(feat), dim=1)
        pooled = torch.sum(feat * weights, dim=1)
        return pooled


class MRIAttentionClassifier(nn.Module):
    """Backbone ConvNeXt-Tiny đơn mặt phẳng thích ứng kênh ảnh y tế 1-channel[cite: 2]."""
    def __init__(self, num_classes: int = 12, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # Chuyển đổi conv đầu vào thành 1-kênh (mean weights từ 3 kênh gốc)
        stem_conv = backbone.features[0][0]
        out_channels = stem_conv.out_channels
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=4, stride=4)
        if pretrained:
            self.conv1.weight.data = stem_conv.weight.data.mean(dim=1, keepdim=True)
            if stem_conv.bias is not None:
                self.conv1.bias.data = stem_conv.bias.data.clone()

        backbone.features[0][0] = self.conv1
        self.encoder = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = backbone.classifier[2].in_features

        # Vị trí lát cắt và khối gom cụm Transformer
        self.pos_embedding = nn.Embedding(96, self.feature_dim)
        self.aggregator = SliceTransformerAggregation(embed_dim=self.feature_dim, num_heads=4, num_layers=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 1, S, H, W]
        if x.shape[1] == 1:
            x = x.permute(0, 2, 1, 3, 4)

        B, S, C, H, W = x.shape
        x = x.reshape(B * S, C, H, W)

        feat = self.encoder(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)
        feat = feat.view(B, S, -1)

        pos_ids = torch.arange(S, device=feat.device).unsqueeze(0).expand(B, S)
        feat = feat + self.pos_embedding(pos_ids)

        volume_feat = self.aggregator(feat)
        return volume_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)