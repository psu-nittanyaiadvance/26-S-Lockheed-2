from __future__ import annotations

from typing import Any, Iterable, Optional

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


def _detect_modality(meta: dict) -> str:
    """Infer modality from metadata; defaults to 'sar' for backwards compat."""
    if "fused_optical_available" in meta:
        return "fused"
    return str(meta.get("modality", "sar"))


def validate_sample_shapes(ds: Any, n: int = 64) -> None:
    """
    Validate sample shapes and dtypes for the first n items of a dataset.

    Works with SARDataset, OpticalDataset, and FusedDataset.

    SAR checks
    ----------
    - image is [C,H,W], dtype float32, finite values only.
    - mask is [1,H,W], dtype uint8, binary {0,1}.

    Optical checks
    --------------
    - image is [C,H,W], dtype float32, finite values only.
    - image values are in [0, 1] after normalisation (if normalize_cfg="none")
      or unbounded (zscore).  Only the [0,1] clamp is checked when modality
      metadata indicates "optical" and no zscore normalisation is in use.
    - mask is [1,H,W], dtype uint8, binary {0,1} (same as SAR).

    Fused checks
    ------------
    - image is [C_sar+C_s2, H, W], dtype float32, finite.
    - n_sar_bands + n_optical_bands == C dimension.
    - mask is [1,H,W], dtype uint8, binary {0,1}.
    """
    total = len(ds)
    _require(total > 0, "Dataset is empty")

    for idx in _iter_indices(n, total):
        img, mask, meta = ds[idx]
        sample_id = meta.get("id", f"index_{idx}")
        modality = _detect_modality(meta)

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

        # Modality-specific checks
        if modality == "optical":
            _validate_optical_image(img_np, sample_id, ds)

        elif modality == "fused":
            n_sar = meta.get("n_sar_bands")
            n_opt = meta.get("n_optical_bands")
            if n_sar is not None and n_opt is not None:
                _require(
                    img_np.shape[0] == n_sar + n_opt,
                    f"Fused channel count mismatch for id '{sample_id}': "
                    f"expected {n_sar}+{n_opt}={n_sar + n_opt} bands, "
                    f"got {img_np.shape[0]}",
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


def _validate_optical_image(img_np: np.ndarray, sample_id: str, ds: Any) -> None:
    """
    Additional checks specific to OpticalDataset samples.

    Only checks the [0,1] value range when the dataset uses normalize_cfg="none"
    (raw reflectance).  After zscore normalisation values may be unbounded.
    """
    normalize_type = getattr(ds, "_normalize_type", None) or getattr(
        getattr(ds, "optical_dataset", None), "_normalize_type", None
    )
    if normalize_type == "none" or normalize_type is None:
        lo, hi = float(img_np.min()), float(img_np.max())
        _require(
            lo >= -1e-4 and hi <= 1.0 + 1e-4,
            f"Optical image values outside [0,1] for id '{sample_id}': "
            f"min={lo:.4f}, max={hi:.4f}. "
            "Ensure scale-to-reflectance and clipping ran correctly.",
        )
