import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.pseudo_labeling import LABEL_COLS
from src.utils import DriveCheckpointManager, compute_macro_auc
from src.vision_models import (
    MRIStudyFusionClassifier,
    MaskedAsymmetricLoss,
    StudyFusionVolumeDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Mid-Level Cross-Plane Fusion (Giai đoạn 2).")
    parser.add_argument("--fusion_csv", type=str, default="data/processed/df_study_fusion.csv", help="Đường dẫn file df_study_fusion.csv.")
    parser.add_argument("--volumes_dir", type=str, default="/content/knee_volumes", help="Thư mục chứa các file .npz.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Thư mục chứa/lưu trữ checkpoints.")
    parser.add_argument("--epochs", type=int, default=6, help="Số epoch tối đa cho fusion.")
    parser.add_argument("--batch_size", type=int, default=2, help="Kích thước batch cho fusion.")
    parser.add_argument("--patience", type=int, default=2, help="Số epoch kiên nhẫn trước khi dừng sớm (Early Stopping).")
    parser.add_argument("--use_drive", action="store_true", help="Kích hoạt đồng bộ hóa qua Google Drive.")
    return parser.parse_args()


def train_fusion_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None):
    model.train()
    running_loss, total = 0.0, 0
    for (sag, cor, axi), labels in loader:
        sag, cor, axi = sag.to(device, non_blocking=True), cor.to(device, non_blocking=True), axi.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        b_size = labels.size(0)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(sag, cor, axi)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(sag, cor, axi)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * b_size
        total += b_size
    return running_loss / max(1, total)


@torch.no_grad()
def eval_fusion_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for (sag, cor, axi), labels in loader:
        sag, cor, axi = sag.to(device, non_blocking=True), cor.to(device, non_blocking=True), axi.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        b_size = labels.size(0)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(sag, cor, axi)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * b_size
        total += b_size
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / max(1, total)
    if not all_preds:
        return avg_loss, 0.5
    macro_auc = compute_macro_auc(np.concatenate(all_targets, axis=0), np.concatenate(all_preds, axis=0))
    return avg_loss, macro_auc


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    df_fusion = pd.read_csv(args.fusion_csv)
    for c in LABEL_COLS:
        if c in df_fusion.columns:
            df_fusion[c] = pd.to_numeric(df_fusion[c], errors="coerce").fillna(-1.0).astype(np.float32)

    # Group K-Fold theo ca chụp để đảm bảo tính khách quan
    unique_studies = df_fusion["StudyInstanceUID"].drop_duplicates().values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(kf.split(unique_studies))

    train_studies = set(unique_studies[tr_idx])
    val_studies = set(unique_studies[va_idx])

    train_df = df_fusion[df_fusion["StudyInstanceUID"].isin(train_studies)].reset_index(drop=True)
    val_df = df_fusion[df_fusion["StudyInstanceUID"].isin(val_studies)].reset_index(drop=True)

    train_ds = StudyFusionVolumeDataset(train_df, args.volumes_dir, LABEL_COLS, is_train=True)
    val_ds = StudyFusionVolumeDataset(val_df, args.volumes_dir, LABEL_COLS, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=is_cuda)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=is_cuda)

    # Chuẩn bị backbone pre-trained
    pretrained_backbones = {
        "sag": os.path.join(args.checkpoint_dir, "best_sagittal_backbone.pth"),
        "cor": os.path.join(args.checkpoint_dir, "best_coronal_backbone.pth"),
        "axi": os.path.join(args.checkpoint_dir, "best_axial_backbone.pth")
    }

    model = MRIStudyFusionClassifier(
        num_classes=len(LABEL_COLS),
        pretrained_backbones=pretrained_backbones,
        freeze_backbones=True
    ).to(device)

    criterion = MaskedAsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1.2e-4, weight_decay=3e-2)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.2e-4, total_steps=total_steps, pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda) if is_cuda else None

    drive_mgr = DriveCheckpointManager() if args.use_drive else None
    best_ckpt_name = "best_mri_fusion_model.pth"
    best_auc = 0.0
    no_improve = 0

    print(f"\n>>> Bắt đầu huấn luyện Study Fusion Model ({args.epochs} Epochs) <<<")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_fusion_epoch(model, train_loader, criterion, optimizer, scaler, device, scheduler)
        va_loss, va_auc = eval_fusion_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Fusion Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Val AUC: {va_auc:.4f} ({elapsed:.1f}s)")

        if va_auc > best_auc:
            best_auc = va_auc
            no_improve = 0
            save_payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_auc": best_auc}
            local_path = os.path.join(args.checkpoint_dir, best_ckpt_name)
            torch.save(save_payload, local_path)
            if drive_mgr:
                drive_mgr.update_checkpoint(save_payload, best_ckpt_name, local_dir=args.checkpoint_dir)
            print(f"  --> Cập nhật kỷ lục AUC mới: {best_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"🛑 [Early Stopping] Dừng tại Epoch {epoch:02d} do không cải thiện sau {args.patience} epochs.")
                break

    print(f"✅ Hoàn tất huấn luyện Fusion! Best Macro AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()