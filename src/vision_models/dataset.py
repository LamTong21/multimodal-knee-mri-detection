import glob
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def load_processed_volume(
    series_id: str,
    data_dir: str,
    num_slices: int = 24,
    img_size: tuple = (224, 224)
) -> torch.Tensor:
    """Nạp file .npz và nội suy 3D Trilinear thành kích thước chuẩn [1, S, H, W][cite: 2]."""
    if pd.isna(series_id) or not str(series_id).strip():
        return torch.zeros((1, num_slices, img_size[0], img_size[1]), dtype=torch.float32)

    series_id = str(series_id).strip()
    npz_path = os.path.join(data_dir, f"{series_id}.npz")

    if not os.path.exists(npz_path):
        matches = glob.glob(os.path.join(data_dir, "**", f"{series_id}.npz"), recursive=True)
        if matches:
            npz_path = matches[0]
        else:
            return torch.zeros((1, num_slices, img_size[0], img_size[1]), dtype=torch.float32)

    try:
        with np.load(npz_path) as data:
            key = 'arr_0' if 'arr_0' in data else list(data.files)[0]
            arr = data[key].astype(np.float32)

        if arr.max() > 1.0:
            arr /= 255.0

        if arr.ndim == 3 and arr.shape[0] > arr.shape[2]:
            arr = np.transpose(arr, (2, 0, 1))

        tensor_vol = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        if tensor_vol.shape[2:] != (num_slices, img_size[0], img_size[1]):
            tensor_vol = F.interpolate(
                tensor_vol,
                size=(num_slices, img_size[0], img_size[1]),
                mode='trilinear',
                align_corners=False
            )
        return tensor_vol.squeeze(0)
    except Exception as e:
        print(f"Lỗi khi đọc file thể tích series {series_id}: {e}")
        return torch.zeros((1, num_slices, img_size[0], img_size[1]), dtype=torch.float32)


class MRI3DTransform:
    """Tập tăng cường dữ liệu affine 3D đồng nhất trên toàn bộ lát cắt."""
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if not self.is_train:
            return volume

        # 1. Flip ngang
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[3])

        # 2. Random Affine xoay & trượt đồng bộ các lát cắt
        if random.random() > 0.5:
            angle = random.uniform(-10.0, 10.0)
            trans_x = int(0.05 * volume.shape[3])
            trans_y = int(0.05 * volume.shape[2])
            translations = (random.randint(-trans_x, trans_x), random.randint(-trans_y, trans_y))
            scale = random.uniform(0.95, 1.05)

            slices = [
                TF.affine(volume[:, s, :, :], angle=angle, translate=translations, scale=scale, shear=0)
                for s in range(volume.shape[1])
            ]
            volume = torch.stack(slices, dim=1)

        # 3. Random Contrast / Gamma
        if random.random() > 0.5:
            gamma = random.uniform(0.85, 1.15)
            volume = torch.clamp(volume ** gamma, 0.0, 1.0)

        return volume


class SeriesMRIVolumeDataset(Dataset):
    """Dataset cho Giai đoạn 1: Huấn luyện đơn mặt phẳng (Sagittal, Coronal, hoặc Axial)[cite: 2]."""
    def __init__(self, df: pd.DataFrame, data_dir: str, label_cols: list = None, num_slices: int = 24, img_size=(224, 224), is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.num_slices = num_slices
        self.img_size = img_size
        self.transform = MRI3DTransform(is_train=is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = str(row["SeriesInstanceUID"])
        volume = load_processed_volume(series_id, self.data_dir, self.num_slices, self.img_size)
        volume = self.transform(volume)

        if self.label_cols:
            labels = torch.tensor(row[self.label_cols].fillna(-1.0).values.astype(np.float32), dtype=torch.float32)
            return volume, labels
        return volume, str(row.get("StudyInstanceUID", series_id))


class StudyFusionVolumeDataset(Dataset):
    """Dataset cho Giai đoạn 2: Ghép bộ ba mặt phẳng (Sagittal, Coronal, Axial) theo StudyInstanceUID[cite: 2]."""
    def __init__(self, df: pd.DataFrame, data_dir: str, label_cols: list = None, num_slices: int = 24, img_size=(224, 224), is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.num_slices = num_slices
        self.img_size = img_size
        self.transform = MRI3DTransform(is_train=is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row["StudyInstanceUID"])

        sag_id = row.get("SeriesInstanceUID_sag", row.get("sagittal_series_uid"))
        cor_id = row.get("SeriesInstanceUID_cor", row.get("coronal_series_uid"))
        axi_id = row.get("SeriesInstanceUID_axi", row.get("axial_series_uid"))

        vol_sag = self.transform(load_processed_volume(sag_id, self.data_dir, self.num_slices, self.img_size))
        vol_cor = self.transform(load_processed_volume(cor_id, self.data_dir, self.num_slices, self.img_size))
        vol_axi = self.transform(load_processed_volume(axi_id, self.data_dir, self.num_slices, self.img_size))

        inputs = (vol_sag, vol_cor, vol_axi)
        if self.label_cols:
            labels = torch.tensor(row[self.label_cols].fillna(-1.0).values.astype(np.float32), dtype=torch.float32)
            return inputs, labels
        return inputs, study_id