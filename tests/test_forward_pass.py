import pytest
import torch

from src.pseudo_labeling.prompt_templates import LABEL_COLS
from src.vision_models.backbones import MRIAttentionClassifier, SliceTransformerAggregation
from src.vision_models.fusion_net import MRIStudyFusionClassifier
from src.vision_models.loss import MaskedAsymmetricLoss


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_slice_transformer_aggregation():
    """Kiểm tra chiều không gian của lớp SliceTransformerAggregation."""
    batch_size = 2
    num_slices = 24
    embed_dim = 768

    aggregator = SliceTransformerAggregation(embed_dim=embed_dim, num_heads=4, num_layers=2)
    dummy_features = torch.randn(batch_size, num_slices, embed_dim)

    output = aggregator(dummy_features)
    assert output.shape == (batch_size, embed_dim), (
        f"Kỳ vọng kích thước [2, 768] nhưng nhận được {list(output.shape)}"
    )


def test_mri_attention_classifier_forward(device):
    """Kiểm tra chu trình chuyển tiếp (forward pass) của Backbone đơn mặt phẳng ConvNeXt-Tiny[cite: 2]."""
    batch_size = 2
    num_slices = 16  # Dùng 16 lát cắt để chạy test nhanh trên CPU
    h, w = 224, 224

    model = MRIAttentionClassifier(num_classes=len(LABEL_COLS), pretrained=False).to(device)
    # Tensor thể tích MRI đầu vào: [Batch, Channel=1, Slices, H, W][cite: 2]
    dummy_volume = torch.randn(batch_size, 1, num_slices, h, w, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_volume)

    assert logits.shape == (batch_size, 12), (
        f"Kỳ vọng kích thước logits [2, 12] nhưng nhận được {list(logits.shape)}"
    )
    assert not torch.isnan(logits).any(), "Phát hiện giá trị NaN trong đầu ra của Backbone"


def test_mri_study_fusion_classifier_forward(device):
    """Kiểm tra kiến trúc Mid-Level Cross-Plane Attention Fusion với 3 thể tích mặt phẳng[cite: 2]."""
    batch_size = 2
    num_slices = 16
    h, w = 224, 224

    # Khởi tạo mô hình fusion không tải pre-trained để test nhanh
    model = MRIStudyFusionClassifier(num_classes=len(LABEL_COLS), pretrained_backbones=None, freeze_backbones=False).to(device)

    sag_vol = torch.randn(batch_size, 1, num_slices, h, w, device=device)
    cor_vol = torch.randn(batch_size, 1, num_slices, h, w, device=device)
    axi_vol = torch.randn(batch_size, 1, num_slices, h, w, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(sag_vol, cor_vol, axi_vol)

    assert logits.shape == (batch_size, 12), (
        f"Kỳ vọng kích thước logits [2, 12] nhưng nhận được {list(logits.shape)}"
    )
    assert not torch.isnan(logits).any(), "Phát hiện giá trị NaN trong đầu ra của Fusion Network"


def test_masked_asymmetric_loss_gradient_flow():
    """Kiểm tra Masked Asymmetric Loss xử lý đúng các nhãn không xác định (-1, NaN) và tính gradient chuẩn xác."""
    criterion = MaskedAsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    
    # 2 mẫu, 4 nhãn
    logits = torch.randn(2, 4, requires_grad=True)
    targets = torch.tensor([
        [1.0, 0.0, -1.0, 1.0],  # Nhãn vị trí 2 không xác định (-1)
        [0.0, 1.0, float('nan'), 0.0]  # Nhãn vị trí 2 là NaN
    ], dtype=torch.float32)

    loss = criterion(logits, targets)
    assert not torch.isnan(loss), "Hàm loss trả về giá trị NaN khi có nhãn rỗng"
    assert loss.item() > 0.0, "Giá trị loss phải dương"

    loss.backward()
    assert logits.grad is not None, "Gradient không được truyền ngược"
    # Gradient tại vị trí nhãn bị mask (-1 và NaN) phải triệt tiêu về 0
    assert logits.grad[0, 2] == 0.0, "Vị trí nhãn -1 không được tính gradient"
    assert logits.grad[1, 2] == 0.0, "Vị trí nhãn NaN không được tính gradient"