from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union
import warnings

import numpy as np
import pandas as pd
import torch

from .sar_dataset import TIME_MATCHED_ZEROS_WARNING

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
    - mask is [1,H,W] when present, dtype uint8, binary {0,1}
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


def validate_time_matched(ds: Any, n: int = 64) -> None:
    """
    Validate time-matched stacks when ds.use_time_matched is enabled.

    Checks:
    - time_matched is [B,H,W] when present, dtype float32 (B = expected_time_matched_bands)
    - missing policy behavior:
      - zeros: missing stacks return zeros (warning only once)
      - skip: missing stacks are filtered
      - raise: missing stacks raise at access
    """
    if not getattr(ds, "use_time_matched", False):
        return

    total = len(ds)
    _require(total > 0, "Dataset is empty")

    policy = getattr(ds, "time_matched_missing_policy", "zeros")
    expected_bands = int(getattr(ds, "expected_time_matched_bands", 8))
    missing_statuses = {"missing", "missing_s1", "missing_s2", "missing_both"}

    missing_idx: Optional[int] = None
    saw_missing = False

    for idx in _iter_indices(n, total):
        try:
            img, _, meta = ds[idx]
        except RuntimeError as exc:
            if policy == "raise" and "Missing time-matched stack" in str(exc):
                saw_missing = True
                missing_idx = idx
                continue
            raise

        sample_id = meta.get("id", f"index_{idx}")
        tm = meta.get("time_matched")
        tm_status = str(meta.get("time_matched_status") or "ok")

        if tm is None:
            if policy == "raise":
                raise ValueError(
                    f"time_matched missing in metadata for id '{sample_id}' while policy='raise'"
                )
            raise ValueError(f"time_matched missing in metadata for id '{sample_id}'")

        tm_np = _as_numpy(tm)
        _require(
            tm_np.ndim == 3 and tm_np.shape[0] == expected_bands,
            f"time_matched must be [{expected_bands},H,W] for id '{sample_id}', got {tm_np.shape}",
        )
        _require(
            tm_np.dtype == np.float32,
            f"time_matched dtype must be float32 for id '{sample_id}', got {tm_np.dtype}",
        )

        if tm_status in missing_statuses:
            saw_missing = True
            missing_idx = idx
            if policy == "zeros":
                _require(
                    np.all(tm_np == 0),
                    f"time_matched should be zeros for missing id '{sample_id}', "
                    f"status={tm_status}",
                )
            elif policy == "skip":
                raise ValueError(
                    f"Found missing time_matched sample with policy='skip' for id '{sample_id}'"
                )
            elif policy == "raise":
                raise ValueError(
                    f"Missing time_matched sample returned without raising for id '{sample_id}'"
                )

    if policy == "skip":
        if hasattr(ds, "samples") and hasattr(ds, "_time_matched_path"):
            for idx in _iter_indices(n, total):
                sample = ds.samples[idx]
                tm_path, tm_status = ds._time_matched_path(sample)
                if tm_status and tm_status != "ok":
                    raise ValueError(
                        f"Policy='skip' should filter missing status {tm_status} for id '{sample['id']}'"
                    )
                if tm_path is None:
                    raise ValueError(
                        f"Policy='skip' should filter missing time_matched for id '{sample['id']}'"
                    )

    if policy == "raise" and not saw_missing:
        warnings.warn("No missing time_matched samples encountered; policy='raise' not exercised.")

    if policy == "zeros" and missing_idx is not None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = ds[missing_idx]
            _ = ds[missing_idx]
        warn_count = sum(
            1 for w in caught if str(w.message) == TIME_MATCHED_ZEROS_WARNING
        )
        if warn_count > 1:
            raise ValueError("Expected a single warning for missing time-matched zeros policy.")


def validate_manifest_consistency(manifest_path: Union[str, Path]) -> None:
    """
    Validate time-matched manifest consistency.

    Required columns: sample_id, path, status, s1_date_used, s2_date_used, window_s1, window_s2
    Allowed status values: ok, missing_s1, missing_s2, missing_both, error
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    required = {"sample_id", "path", "status", "s1_date_used", "s2_date_used", "window_s1", "window_s2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    allowed = {"ok", "missing_s1", "missing_s2", "missing_both", "error"}
    bad = sorted({str(s) for s in df["status"].dropna().unique()} - allowed)
    if bad:
        raise ValueError(f"Manifest contains invalid status values: {bad}")
