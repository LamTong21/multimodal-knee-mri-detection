import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def rank_normalize(matrix: np.ndarray) -> np.ndarray:
    """Chuyển đổi xác suất theo từng nhãn thành rank percentile [0, 1] cho Macro AUC[cite: 2]."""
    ranked = np.zeros_like(matrix)
    for col in range(matrix.shape[1]):
        r = rankdata(matrix[:, col], method="average")
        denom = max(1, len(r) - 1)
        ranked[:, col] = (r - 1.0) / denom
    return ranked


def compute_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính trung bình Macro AUC ROC cho các nhãn hợp lệ (loại bỏ -1 hoặc NaN)[cite: 2]."""
    auc_scores = []
    for col_idx in range(y_true.shape[1]):
        t = y_true[:, col_idx]
        p = y_pred[:, col_idx]

        valid = (t >= 0.0) & (~np.isnan(t))
        t_valid = t[valid]
        p_valid = p[valid]

        if len(np.unique(t_valid)) > 1:
            try:
                auc_scores.append(roc_auc_score(t_valid, p_valid))
            except Exception:
                pass
    return float(np.mean(auc_scores)) if auc_scores else 0.5


def compute_pos_weights(df: pd.DataFrame, label_cols: list) -> torch.Tensor:
    """Tính trọng số nghịch đảo mất cân bằng nhãn: neg_count / pos_count[cite: 2]."""
    weights = []
    total_samples = len(df)
    for col in label_cols:
        if col in df.columns:
            pos_count = (df[col] == 1.0).sum()
            w = (total_samples - pos_count) / max(1, pos_count)
            weights.append(min(10.0, max(1.0, float(w))))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)