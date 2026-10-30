import numpy as np
import pandas as pd
import pytest

from src.pseudo_labeling.milp_selector import select_cases_ilp
from src.pseudo_labeling.prompt_templates import LABEL_COLS


@pytest.fixture
def dummy_labeled_data():
    """Tạo bảng giả lập 50 ca bệnh với phân bổ nhãn thưa và mất cân bằng."""
    np.random.seed(42)
    n_samples = 50
    data = {"StudyInstanceUID": [f"UID_{i:03d}" for i in range(n_samples)]}

    for col in LABEL_COLS:
        # Xác suất xuất hiện dương tính phân bổ ngẫu nhiên từ 5% đến 25%
        prob = np.random.uniform(0.05, 0.25)
        data[col] = np.random.choice([0, 1], size=n_samples, p=[1.0 - prob, prob])

    # Đảm bảo mỗi nhãn xuất hiện ít nhất một lần trong toàn bộ tập
    df = pd.DataFrame(data)
    for col in LABEL_COLS:
        if df[col].sum() == 0:
            df.loc[np.random.randint(0, n_samples), col] = 1

    return df


def test_milp_exact_case_count(dummy_labeled_data):
    """Kiểm tra số lượng ca bệnh được chọn phải khớp chính xác với chỉ tiêu k."""
    k_target = 6
    selected_df = select_cases_ilp(dummy_labeled_data, label_cols=LABEL_COLS, k=k_target, min_pos_per_label=1)
    
    assert len(selected_df) == k_target, f"Kỳ vọng {k_target} ca nhưng nhận được {len(selected_df)}"
    assert selected_df["StudyInstanceUID"].is_unique, "Các ca được chọn không được trùng lặp"


def test_milp_full_label_coverage(dummy_labeled_data):
    """Đảm bảo mọi nhãn trong 12 điều kiện đều có tối thiểu min_pos_per_label ca dương tính[cite: 1]."""
    min_pos = 1
    selected_df = select_cases_ilp(dummy_labeled_data, label_cols=LABEL_COLS, k=6, min_pos_per_label=min_pos)

    label_sums = selected_df[LABEL_COLS].sum(axis=0)
    for col in LABEL_COLS:
        assert label_sums[col] >= min_pos, (
            f"Nhãn '{col}' không đạt yêu cầu tối thiểu {min_pos} ca dương tính (thực tế: {label_sums[col]})"
        )


def test_milp_adaptive_relaxation(dummy_labeled_data):
    """Kiểm tra cơ chế tự động mở rộng k (fallback relaxation) khi bài toán infeasible với k quá nhỏ[cite: 1]."""
    # Ép một nhãn chỉ xuất hiện đúng 1 lần trên toàn tập và k ban đầu đặt bằng 1 (chắc chắn không phủ hết 12 nhãn)
    dummy_labeled_data.loc[:, "Fracture"] = 0
    dummy_labeled_data.loc[0, "Fracture"] = 1

    selected_df = select_cases_ilp(dummy_labeled_data, label_cols=LABEL_COLS, k=2, min_pos_per_label=1)
    
    # Giải thuật sẽ đệ quy tăng dần k (+2) cho đến khi tìm được lời giải khả thi
    assert len(selected_df) >= 2
    assert selected_df["Fracture"].sum() >= 1