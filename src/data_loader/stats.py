from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .sar_dataset import SARDataset

def compute_running_mean_std(
    dataset: Dataset,
    max_samples: Optional[int] = None,
    require_no_transforms: bool = True,
    require_normalize_none: bool = True,
    include_log_transform_flag: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-band mean/std using Welford's algorithm (streaming).

    The dataset is expected to return (image, mask, metadata) and images as [C,H,W].
    For raw statistics, initialize dataset with normalize_cfg="none" and transforms=None.

    IMPORTANT: compute statistics on the TRAIN split only to avoid leakage.
    Guardrails can be disabled via require_no_transforms/require_normalize_none or
    include_log_transform_flag if you intentionally need different settings.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if require_no_transforms and getattr(dataset, "transforms", None) is not None:
        raise ValueError(
            "Transforms are enabled on the dataset. Compute stats on raw data by "
            "setting transforms=None or pass require_no_transforms=False."
        )

    if isinstance(dataset, SARDataset):
        if require_normalize_none and getattr(dataset, "_normalize_type", "none") != "none":
            raise ValueError(
                "Normalization is enabled on the dataset. Compute stats before normalization "
                "by setting normalize_cfg='none' or pass require_normalize_none=False."
            )
        if include_log_transform_flag is not None:
            if bool(getattr(dataset, "log_transform", False)) != bool(include_log_transform_flag):
                raise ValueError(
                    "Dataset log_transform does not match include_log_transform_flag. "
                    "Compute stats with matching log transform settings."
                )

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
