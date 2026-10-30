import numpy as np
import pandas as pd
import pytest

from scripts.02_prepare_fusion_tables import build_fusion_dataframe
from src.pseudo_labeling.prompt_templates import LABEL_COLS


@pytest.fixture
def sample_series_and_labels():
    """Tạo metadata mô phỏng các series MRI và nhãn chẩn đoán[cite: 2]."""
    study_ids = [f"Study_{i}" for i in range(10)]
    planes = ["Sagittal", "Coronal", "Axial"]
    
    series_records = []
    for s_id in study_ids:
        for p in planes:
            # Mô phỏng trường hợp mỗi mặt phẳng có thể có chuỗi ảnh có hoặc không có xóa mỡ (Fat_Suppression)
            series_records.append({
                "StudyInstanceUID": s_id,
                "SeriesInstanceUID": f"{s_id}_{p[:3]}_noFS",
                "Anatomical_Plane": p,
                "Fat_Suppression": 0
            })
            series_records.append({
                "StudyInstanceUID": s_id,
                "SeriesInstanceUID": f"{s_id}_{p[:3]}_FS",
                "Anatomical_Plane": p,
                "Fat_Suppression": 1
            })

    series_df = pd.DataFrame(series_records)

    labels_data = {"StudyInstanceUID": study_ids}
    for col in LABEL_COLS:
        labels_data[col] = np.random.choice([0, 1], size=len(study_ids))
    labels_df = pd.DataFrame(labels_data)

    return series_df, labels_df, planes


def test_fusion_prioritizes_fat_suppression(sample_series_and_labels):
    """Kiểm tra thuật toán ghép bảng ưu tiên chọn chuỗi chụp có Fat_Suppression=1[cite: 2]."""
    series_df, labels_df, planes = sample_series_and_labels
    fusion_df = build_fusion_dataframe(series_df, planes=planes, labels_df=labels_df, is_test=False)

    for _, row in fusion_df.iterrows():
        assert str(row["SeriesInstanceUID_sag"]).endswith("_FS"), "Chưa ưu tiên chuỗi Fat Suppression mặt Sagittal"
        assert str(row["SeriesInstanceUID_cor"]).endswith("_FS"), "Chưa ưu tiên chuỗi Fat Suppression mặt Coronal"
        assert str(row["SeriesInstanceUID_axi"]).endswith("_FS"), "Chưa ưu tiên chuỗi Fat Suppression mặt Axial"


def test_fusion_preserves_labels_and_study_uids(sample_series_and_labels):
    """Đảm bảo các nhãn chẩn đoán và StudyInstanceUID giữ nguyên giá trị không bị lệch hàng[cite: 2]."""
    series_df, labels_df, planes = sample_series_and_labels
    fusion_df = build_fusion_dataframe(series_df, planes=planes, labels_df=labels_df, is_test=False)

    assert len(fusion_df) == len(labels_df), "Số lượng dòng trong bảng fusion bị sai lệch so với ground truth"
    
    # Kiểm tra tính đồng nhất của từng cột nhãn
    sorted_orig = labels_df.sort_values("StudyInstanceUID").reset_index(drop=True)
    sorted_fused = fusion_df.sort_values("StudyInstanceUID").reset_index(drop=True)

    pd.testing.assert_series_equal(sorted_orig["StudyInstanceUID"], sorted_fused["StudyInstanceUID"])
    for col in LABEL_COLS:
        pd.testing.assert_series_equal(sorted_orig[col], sorted_fused[col], check_names=False)


def test_test_fusion_structure_without_labels(sample_series_and_labels):
    """Kiểm tra cấu trúc ghép chuỗi ảnh cho tập Test (không có nhãn và không bị drop ca thiếu)[cite: 2]."""
    series_df, _, planes = sample_series_and_labels
    test_fusion = build_fusion_dataframe(series_df, planes=planes, labels_df=None, is_test=True)

    expected_cols = {"StudyInstanceUID", "SeriesInstanceUID_sag", "SeriesInstanceUID_cor", "SeriesInstanceUID_axi"}
    assert expected_cols.issubset(set(test_fusion.columns)), "Bảng test fusion thiếu các cột series mặt phẳng bắt buộc"
    for col in LABEL_COLS:
        assert col not in test_fusion.columns, f"Nhãn '{col}' không được phép xuất hiện trong bảng test"