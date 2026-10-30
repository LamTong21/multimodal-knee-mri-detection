import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from .prompt_templates import LABEL_COLS


def select_cases_ilp(
    df: pd.DataFrame,
    label_cols: list = LABEL_COLS,
    k: int = 4,
    min_pos_per_label: int = 1
) -> pd.DataFrame:
    """
    0-1 MILP: Chọn tập nhỏ k ca Few-shot bao phủ toàn diện 12 nhãn, ưu tiên các nhãn hiếm[cite: 1].
    """
    n_samples = len(df)
    A = df[label_cols].to_numpy().astype(int)

    # Ưu tiên các nhãn hiếm thông qua nghịch đảo tần suất xuất hiện
    rarity = 1.0 / (A.sum(axis=0) + 1e-5)
    c = -(A @ rarity)

    integrality = np.ones(n_samples)
    bounds = Bounds(0, 1)

    # Ràng buộc số ca được chọn chính xác là k
    A_eq = np.ones((1, n_samples))
    constraint_k = LinearConstraint(A_eq, lb=k, ub=k)

    # Ràng buộc mỗi nhãn phải xuất hiện ít nhất min_pos_per_label lần
    constraint_cover = LinearConstraint(A.T, lb=min_pos_per_label, ub=np.inf)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=[constraint_k, constraint_cover]
    )

    if res.success:
        selected_idx = np.where(res.x > 0.5)[0]
        selected_df = df.iloc[selected_idx].copy()
        print(f"--> [MILP] Đã giải quyết tối ưu: Chọn {len(selected_df)} ca Few-Shot đại diện.")
        return selected_df
    else:
        print(f"--> [MILP Warning] Không tìm thấy nghiệm với k={k}. Tăng k lên {k+2}...")
        return select_cases_ilp(df, label_cols, k=k+2, min_pos_per_label=min_pos_per_label)