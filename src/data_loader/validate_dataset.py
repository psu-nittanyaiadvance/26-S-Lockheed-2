from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch


def _as_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _iter_indices(n: int, total: int) -> Iterable[int]:
    count = min(n, total)
    return range(count)


def validate_sample_shapes(ds: Any, n: int = 64) -> None:
    """
    Validate sample shapes and dtypes for the first n items of a dataset.

    Checks:
    - image is [C,H,W], dtype float32, finite values only
    - mask is [1,H,W], dtype uint8, binary {0,1}
    """
    total = len(ds)
    _require(total > 0, "Dataset is empty")

    for idx in _iter_indices(n, total):
        img, mask, meta = ds[idx]
        sample_id = meta.get("id", f"index_{idx}")

        img_np = _as_numpy(img)
        _require(
            img_np.ndim == 3,
            f"Image must be [C,H,W] for id '{sample_id}', got shape {img_np.shape}",
        )
        _require(
            img_np.dtype == np.float32,
            f"Image dtype must be float32 for id '{sample_id}', got {img_np.dtype}",
        )
        _require(
            np.isfinite(img_np).all(),
            f"Image contains NaN/Inf for id '{sample_id}'",
        )

        if mask is not None:
            mask_np = _as_numpy(mask)
            _require(
                mask_np.ndim == 3 and mask_np.shape[0] == 1,
                f"Mask must be [1,H,W] for id '{sample_id}', got shape {mask_np.shape}",
            )
            _require(
                mask_np.dtype == np.uint8,
                f"Mask dtype must be uint8 for id '{sample_id}', got {mask_np.dtype}",
            )
            uniq = np.unique(mask_np)
            _require(
                np.all(np.isin(uniq, [0, 1])),
                f"Mask must be binary {{0,1}} for id '{sample_id}', got values {uniq}",
            )
