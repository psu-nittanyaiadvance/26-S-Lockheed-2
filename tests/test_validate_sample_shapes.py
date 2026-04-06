from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import FusedDataset, validate_sample_shapes  # noqa: E402
from data_loader.validate_dataset import _detect_modality  # noqa: E402


class _TupleDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]],
        *,
        normalize_type: Optional[str] = None,
    ) -> None:
        self.items = list(items)
        self.samples = [
            {
                "id": meta["id"],
                "img_path": meta.get("img_path"),
                "mask_path": meta.get("mask_path"),
            }
            for _, _, meta in self.items
        ]
        if normalize_type is not None:
            self._normalize_type = normalize_type

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        return self.items[idx]


def _item(
    *,
    sample_id: str = "tile",
    image: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    if image is None:
        image = torch.ones((2, 2, 2), dtype=torch.float32)
    if mask is None:
        mask = torch.ones((1, 2, 2), dtype=torch.uint8)
    metadata: Dict[str, Any] = {
        "id": sample_id,
        "img_path": f"{sample_id}.tif",
        "mask_path": f"{sample_id}_mask.tif" if mask is not None else None,
        "valid_mask": torch.ones((2, 2), dtype=torch.bool),
    }
    if meta is not None:
        metadata.update(meta)
    return image, mask, metadata


def test_rejects_empty_dataset() -> None:
    ds = _TupleDataset([])

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_sample_shapes(ds)


def test_accepts_valid_optical_sample() -> None:
    img = torch.full((3, 2, 2), 0.5, dtype=torch.float32)
    ds = _TupleDataset(
        [_item(image=img, meta={"modality": "optical"})],
        normalize_type="none",
    )

    validate_sample_shapes(ds, n=1)


def test_accepts_valid_fused_sample() -> None:
    sar_ds = _TupleDataset([_item(sample_id="tile", image=torch.ones((2, 2, 2), dtype=torch.float32))])
    opt_ds = _TupleDataset(
        [_item(sample_id="tile", image=torch.ones((3, 2, 2), dtype=torch.float32), mask=None, meta={"modality": "optical"})],
        normalize_type="none",
    )
    fused = FusedDataset(sar_ds, opt_ds)

    validate_sample_shapes(fused, n=1)


def test_accepts_valid_unlabeled_fused_sample() -> None:
    sar_ds = _TupleDataset([_item(sample_id="tile", image=torch.ones((2, 2, 2), dtype=torch.float32), mask=None)])
    opt_ds = _TupleDataset(
        [_item(sample_id="tile", image=torch.ones((3, 2, 2), dtype=torch.float32), mask=None, meta={"modality": "optical"})],
        normalize_type="none",
    )
    fused = FusedDataset(sar_ds, opt_ds)

    validate_sample_shapes(fused, n=1)


def test_rejects_non_float32_image() -> None:
    ds = _TupleDataset([_item(image=torch.ones((2, 2, 2), dtype=torch.float64))])

    with pytest.raises(ValueError, match="Image dtype must be float32"):
        validate_sample_shapes(ds, n=1)


def test_rejects_nan_or_inf_in_image() -> None:
    img = torch.ones((2, 2, 2), dtype=torch.float32)
    img[0, 0, 0] = float("nan")
    ds = _TupleDataset([_item(image=img)])

    with pytest.raises(ValueError, match="Image contains NaN/Inf"):
        validate_sample_shapes(ds, n=1)


def test_rejects_non_uint8_mask() -> None:
    ds = _TupleDataset([_item(mask=torch.ones((1, 2, 2), dtype=torch.int64))])

    with pytest.raises(ValueError, match="Mask dtype must be uint8"):
        validate_sample_shapes(ds, n=1)


def test_rejects_nonbinary_mask() -> None:
    mask = torch.tensor([[[0, 2], [1, 0]]], dtype=torch.uint8)
    ds = _TupleDataset([_item(mask=mask)])

    with pytest.raises(ValueError, match="Mask must be binary"):
        validate_sample_shapes(ds, n=1)


def test_rejects_bad_fused_channel_count() -> None:
    meta = {"fused_optical_available": True, "n_sar_bands": 1, "n_optical_bands": 1}
    ds = _TupleDataset([_item(image=torch.ones((3, 2, 2), dtype=torch.float32), meta=meta)])

    with pytest.raises(ValueError, match="Fused channel count mismatch"):
        validate_sample_shapes(ds, n=1)


def test_optical_none_normalization_enforces_zero_to_one_range() -> None:
    img = torch.full((3, 2, 2), 1.2, dtype=torch.float32)
    ds = _TupleDataset(
        [_item(image=img, meta={"modality": "optical"})],
        normalize_type="none",
    )

    with pytest.raises(ValueError, match="outside \\[0,1\\]"):
        validate_sample_shapes(ds, n=1)


def test_optical_zscore_does_not_enforce_zero_to_one_range() -> None:
    img = torch.full((3, 2, 2), 5.0, dtype=torch.float32)
    ds = _TupleDataset(
        [_item(image=img, meta={"modality": "optical"})],
        normalize_type="zscore",
    )

    validate_sample_shapes(ds, n=1)


def test_detect_modality_routes_optical_fused_and_sar_default() -> None:
    assert _detect_modality({"modality": "optical"}) == "optical"
    assert _detect_modality({"modality": "optical", "fused_optical_available": False}) == "fused"
    assert _detect_modality({}) == "sar"
