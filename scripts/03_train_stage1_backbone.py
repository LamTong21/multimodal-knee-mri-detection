import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.pseudo_labeling import LABEL_COLS
from src.utils import DriveCheckpointManager, compute_macro_auc
from src.vision_models import (
    MRIAttentionClassifier,
    MaskedAsymmetricLoss,
    SeriesMRIVolumeDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình đơn mặt phẳng (Giai đoạn 1).")
    parser.add_argument("--plane", type=str, choices=["sagittal", "coronal", "axial"], required=True, help="Mặt phẳng MRI cần huấn luyện.")
    parser.add_argument("--data_csv", type=str, default=None, help="Đường dẫn file CSV mặt phẳng (mặc định lấy từ data/processed/df_{plane}.csv).")
    parser.add_argument("--volumes_dir", type=str, default="/content/knee_volumes", help="Thư mục chứa các file thể tích .npz.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Thư mục lưu trữ model checkpoints.")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện.")
    parser.add_argument("--batch_size", type=int, default=6, help="Kích thước batch.")
    parser.add_argument("--num_slices", type=int, default=24, help="Số lát cắt MRI 3D.")
    parser.add_argument("--use_drive", action="store_true", help="Lưu trữ/đồng bộ checkpoint lên Google Drive.")
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None):
    model.train()
    running_loss, total = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        b_size = labels.size(0)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
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
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        b_size = labels.size(0)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)
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

    data_csv = args.data_csv or f"data/processed/df_{args.plane}.csv"
    df = pd.read_csv(data_csv)

    # Chia train / validation theo study instance để tránh rò rỉ dữ liệu
    studies = df["StudyInstanceUID"].drop_duplicates().sample(frac=1.0, random_state=42)
    val_sz = max(1, int(len(studies) * 0.2))
    val_set = set(studies.iloc[:val_sz])

    train_df = df[~df["StudyInstanceUID"].isin(val_set)].reset_index(drop=True)
    val_df = df[df["StudyInstanceUID"].isin(val_set)].reset_index(drop=True)

    train_ds = SeriesMRIVolumeDataset(train_df, args.volumes_dir, LABEL_COLS, num_slices=args.num_slices, is_train=True)
    val_ds = SeriesMRIVolumeDataset(val_df, args.volumes_dir, LABEL_COLS, num_slices=args.num_slices, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=is_cuda)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=is_cuda)

    model = MRIAttentionClassifier(num_classes=len(LABEL_COLS), pretrained=True).to(device)
    criterion = MaskedAsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(device)

    # Tối ưu phân tầng (Discriminative learning rate)
    early_conv = list(model.encoder[:4].parameters())
    deep_conv = list(model.encoder[4:].parameters())
    head_params = list(model.pos_embedding.parameters()) + list(model.aggregator.parameters()) + list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": early_conv, "lr": 1e-5},
        {"params": deep_conv, "lr": 4e-5},
        {"params": head_params, "lr": 1.2e-4}
    ], weight_decay=1e-2)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[2e-5, 1e-4, 3e-4], total_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda) if is_cuda else None

    drive_mgr = DriveCheckpointManager() if args.use_drive else None
    best_ckpt_name = f"best_{args.plane}_backbone.pth"
    best_val_auc = 0.0

    print(f"\n>>> Bắt đầu huấn luyện Backbone: {args.plane.upper()} ({args.epochs} Epochs) <<<")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, scheduler)
        va_loss, va_auc = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Val AUC: {va_auc:.4f} ({elapsed:.1f}s)")

        if va_auc > best_val_auc:
            best_val_auc = va_auc
            save_payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_auc": best_val_auc, "plane": args.plane}
            local_path = os.path.join(args.checkpoint_dir, best_ckpt_name)
            torch.save(save_payload, local_path)
            if drive_mgr:
                drive_mgr.update_checkpoint(save_payload, best_ckpt_name, local_dir=args.checkpoint_dir)

    print(f"✅ Hoàn tất Backbone {args.plane}! Kỷ lục Best Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()