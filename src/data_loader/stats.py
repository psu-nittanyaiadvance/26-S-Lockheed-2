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

    Only *valid* (finite) pixels are included in the statistics.  This prevents
    NaN / Inf values — which are imputed to band-min in ``SARDataset`` but
    flagged in ``metadata["valid_mask"]`` — from biasing the normalization
    statistics.

    The dataset is expected to return ``(image, mask, metadata)`` where:
    - ``image`` is a ``[C, H, W]`` tensor (float32).
    - ``metadata`` contains a boolean ``valid_mask`` of shape ``[H, W]``.

    Initialize the dataset with ``normalize_cfg="none"`` and ``transforms=None``
    to obtain raw statistics before applying normalization.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    n: Optional[np.ndarray] = None   # per-band valid pixel counts
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None

    for i in range(total):
        item = dataset[i]
        img, _mask, meta = item

        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)

        if arr.ndim != 3:
            raise ValueError(f"Expected image tensor with shape [C,H,W], got {arr.shape}")

        arr = arr.astype(np.float64)
        c = arr.shape[0]

        # Retrieve the pixel-level validity mask produced by SARDataset.
        # Fall back to a fully-finite mask if the key is absent (e.g. custom datasets).
        vm = meta.get("valid_mask", None)
        if vm is not None:
            if isinstance(vm, torch.Tensor):
                vm = vm.numpy()
            pixel_valid = vm.astype(bool)  # (H, W)
        else:
            # Construct validity from the array itself as a safe fallback.
            pixel_valid = np.isfinite(arr).all(axis=0)  # (H, W)

        if mean is None:
            mean = np.zeros(c, dtype=np.float64)
            M2 = np.zeros(c, dtype=np.float64)
            n = np.zeros(c, dtype=np.int64)

        for band_idx in range(c):
            band = arr[band_idx]
            # Intersect band finiteness with the cross-band validity mask.
            valid_pixels = band[pixel_valid & np.isfinite(band)]
            batch_n = valid_pixels.size
            if batch_n == 0:
                continue

            batch_mean = valid_pixels.mean()
            batch_var = valid_pixels.var()

            # Welford parallel update
            prev_n = n[band_idx]
            total_n = prev_n + batch_n
            delta = batch_mean - mean[band_idx]
            mean[band_idx] += delta * (batch_n / total_n)
            M2[band_idx] += batch_var * batch_n + (delta ** 2) * (prev_n * batch_n / total_n)
            n[band_idx] = total_n

    if mean is None or M2 is None or n is None or n.sum() == 0:
        raise ValueError("No valid (finite) pixels found across the dataset")

    # Guard against bands where every pixel was invalid.
    zero_bands = n == 0
    if zero_bands.any():
        raise ValueError(
            f"Bands {np.where(zero_bands)[0].tolist()} had zero valid pixels — "
            "cannot compute normalization statistics."
        )

    var = M2 / n
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)
