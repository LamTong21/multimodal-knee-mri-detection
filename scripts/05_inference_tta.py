import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.pseudo_labeling import LABEL_COLS
from src.utils import rank_normalize
from src.vision_models import MRIStudyFusionClassifier, StudyFusionVolumeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy inference tập test với TTA và Rank Normalization.")
    parser.add_argument("--test_fusion_csv", type=str, default="data/processed/df_test_fusion.csv", help="Đường dẫn file df_test_fusion.csv.")
    parser.add_argument("--volumes_dir", type=str, default="/content/knee_volumes", help="Thư mục chứa các file .npz.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_mri_fusion_model.pth", help="Trọng số mô hình fusion tốt nhất.")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Đường dẫn lưu file submission.csv.")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch lúc inference.")
    return parser.parse_args()


def predict_tta(model, loader, device):
    model.eval()
    all_study_ids, all_preds = [], []
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for (sag, cor, axi), study_ids in loader:
            sag = sag.to(device, non_blocking=True)
            cor = cor.to(device, non_blocking=True)
            axi = axi.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=is_cuda):
                # 1. Dự đoán bản gốc
                p_orig = torch.sigmoid(model(sag, cor, axi))

                # 2. Test-Time Augmentation: Lật ngang các lát cắt [Batch, Channel, Slices, H, W]
                p_flip = torch.sigmoid(model(
                    torch.flip(sag, dims=[4]),
                    torch.flip(cor, dims=[4]),
                    torch.flip(axi, dims=[4])
                ))
                p_tta = (p_orig + p_flip) / 2.0

            all_study_ids.extend(study_ids)
            all_preds.append(p_tta.cpu().numpy())

    raw_matrix = np.vstack(all_preds)
    return all_study_ids, raw_matrix


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {args.checkpoint}")

    print(f"--> Đang khởi tạo mô hình và nạp trọng số từ: {args.checkpoint}")
    model = MRIStudyFusionClassifier(num_classes=len(LABEL_COLS), freeze_backbones=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(device)

    print(f"--> Nạp dữ liệu test từ: {args.test_fusion_csv}")
    test_df = pd.read_csv(args.test_fusion_csv)
    test_ds = StudyFusionVolumeDataset(test_df, args.volumes_dir, label_cols=None, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("--> Bắt đầu quá trình inference (TTA + Rank Normalization)...")
    study_ids, raw_preds = predict_tta(model, test_loader, device)

    # Chuẩn hóa thứ hạng để tối ưu hóa trực tiếp Macro AUC
    ranked_preds = rank_normalize(raw_preds)

    sub_df = pd.DataFrame(ranked_preds, columns=LABEL_COLS)
    sub_df.insert(0, "StudyInstanceUID", study_ids)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    sub_df.to_csv(args.output_csv, index=False)
    print(f"✅ Đã xuất kết quả submission hoàn tất tại: {args.output_csv} ({len(sub_df)} ca chụp).")


if __name__ == "__main__":
    main()