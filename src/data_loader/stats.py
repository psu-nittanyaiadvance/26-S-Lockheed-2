from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_running_mean_std(
    dataset: Dataset, max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-band mean/std using Welford's algorithm (streaming).

    The dataset is expected to return (image, mask, metadata) and images as [C,H,W].
    For raw statistics, initialize dataset with normalize_cfg="none" and transforms=None.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    n = 0
    mean = None
    M2 = None

    for i in range(total):
        img = dataset[i][0]
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)

        if arr.ndim != 3:
            raise ValueError(f"Expected image tensor with shape [C,H,W], got {arr.shape}")

        arr = arr.astype(np.float64)
        c = arr.shape[0]
        batch_n = arr.shape[1] * arr.shape[2]
        flat = arr.reshape(c, -1)
        batch_mean = flat.mean(axis=1)
        batch_var = flat.var(axis=1)

        if mean is None:
            mean = batch_mean
            M2 = batch_var * batch_n
            n = batch_n
        else:
            delta = batch_mean - mean
            total_n = n + batch_n
            mean = mean + delta * (batch_n / total_n)
            M2 = M2 + batch_var * batch_n + (delta ** 2) * (n * batch_n / total_n)
            n = total_n

    if mean is None or M2 is None or n == 0:
        raise ValueError("No samples found to compute statistics")

    var = M2 / n
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)
