"""
Tiện ích đảm bảo tính tái lập (reproducibility) cho training.

Sửa so với bản cũ: script train gốc KHÔNG hề set seed, khiến mỗi lần chạy
cho kết quả khác nhau — không thể so sánh F1 giữa các lần retrain tự động.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed đầy đủ cho random / numpy / torch / CUDA / PYTHONHASHSEED.

    Gọi trước khi tạo model và DataLoader để mỗi lần chạy cùng một seed
    cho ra cùng một kết quả (trong giới hạn phần cứng).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int) -> None:
    """Seed riêng cho mỗi worker của DataLoader (tránh augmentation/nhiễu lệch nhau)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
