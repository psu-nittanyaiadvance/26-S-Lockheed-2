from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import FusedDataset  # noqa: E402


class _TupleDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]],
        *,
        with_samples: bool = True,
        normalize_type: Optional[str] = None,
    ) -> None:
        self.items = list(items)
        if with_samples:
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


def _make_item(
    sample_id: str,
    *,
    channels: int,
    hw: Tuple[int, int] = (2, 2),
    img_value: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    modality: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    img = torch.full((channels, *hw), img_value, dtype=torch.float32)
    if valid_mask is None:
        valid_mask = torch.ones(hw, dtype=torch.bool)
    meta: Dict[str, Any] = {
        "id": sample_id,
        "img_path": f"{sample_id}.tif",
        "mask_path": f"{sample_id}_mask.tif" if mask is not None else None,
        "valid_mask": valid_mask,
    }
    if modality is not None:
        meta["modality"] = modality
    return img, mask, meta


def test_len_matches_sar_dataset() -> None:
    sar_ds = _TupleDataset(
        [
            _make_item("tile_1", channels=2),
            _make_item("tile_2", channels=2),
        ]
    )
    opt_ds = _TupleDataset([_make_item("tile_1", channels=4, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")

    assert len(fused) == len(sar_ds)


def test_raise_policy_rejects_missing_ids_at_init_when_samples_attr_exists() -> None:
    sar_ds = _TupleDataset([_make_item("sar_only", channels=2)])
    opt_ds = _TupleDataset([_make_item("other_id", channels=4, modality="optical")])

    with pytest.raises(ValueError, match="have no paired optical tile"):
        FusedDataset(sar_ds, opt_ds, optical_missing_policy="raise")


def test_getitem_fuses_channels_and_preserves_mask() -> None:
    sar_mask = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.uint8)
    sar_ds = _TupleDataset(
        [_make_item("tile", channels=2, img_value=1.0, mask=sar_mask)]
    )
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=3, img_value=2.0, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds)
    img, mask, meta = fused[0]

    assert img.shape == (5, 2, 2)
    assert torch.equal(img[:2], sar_ds[0][0])
    assert torch.equal(img[2:], opt_ds[0][0])
    assert mask is not None
    assert torch.equal(mask, sar_mask)
    assert meta["fused_optical_available"] is True
    assert meta["modality"] == "fused"
    assert meta["img_path"] == "tile.tif"
    assert meta["mask_path"] == "tile_mask.tif"


def test_fused_valid_mask_is_logical_and_of_sar_and_optical() -> None:
    sar_valid = torch.tensor([[True, True], [False, True]])
    opt_valid = torch.tensor([[True, False], [True, True]])
    sar_ds = _TupleDataset(
        [_make_item("tile", channels=2, valid_mask=sar_valid)]
    )
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=3, valid_mask=opt_valid, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds)
    _img, _mask, meta = fused[0]

    assert torch.equal(meta["valid_mask"], sar_valid & opt_valid)


def test_missing_optical_with_zeros_policy_zero_fills_and_preserves_sar_valid_mask() -> None:
    sar_mask = torch.ones((1, 2, 2), dtype=torch.uint8)
    sar_valid = torch.tensor([[True, False], [True, True]])
    sar_ds = _TupleDataset(
        [_make_item("missing_tile", channels=2, mask=sar_mask, valid_mask=sar_valid)]
    )
    opt_ds = _TupleDataset(
        [_make_item("other_tile", channels=4, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    img, mask, meta = fused[0]

    assert img.shape == (6, 2, 2)
    assert torch.equal(img[:2], sar_ds[0][0])
    assert torch.equal(img[2:], torch.zeros((4, 2, 2), dtype=torch.float32))
    assert mask is not None
    assert torch.equal(mask, sar_mask)
    assert meta["fused_optical_available"] is False
    assert meta["modality"] == "fused"
    assert meta["img_path"] == "missing_tile.tif"
    assert meta["mask_path"] == "missing_tile_mask.tif"
    assert meta["n_sar_bands"] == 2
    assert meta["n_optical_bands"] == 4
    assert torch.equal(meta["valid_mask"], sar_valid)


def test_missing_optical_with_raise_policy_raises_at_getitem_without_samples_attr() -> None:
    sar_ds = _TupleDataset([_make_item("missing_tile", channels=2)], with_samples=False)
    opt_ds = _TupleDataset([_make_item("other_tile", channels=3, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="raise")

    with pytest.raises(RuntimeError, match="No paired optical tile"):
        _ = fused[0]


def test_spatial_mismatch_raises_when_required() -> None:
    sar_ds = _TupleDataset([_make_item("tile", channels=2, hw=(2, 2))])
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=3, hw=(3, 2), modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds, require_spatial_match=True)

    with pytest.raises(ValueError, match="spatial dims differ"):
        _ = fused[0]


def test_channel_count_metadata_matches_fused_image() -> None:
    sar_ds = _TupleDataset([_make_item("tile", channels=2)])
    opt_ds = _TupleDataset([_make_item("tile", channels=4, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds)
    img, _mask, meta = fused[0]

    assert meta["n_sar_bands"] == 2
    assert meta["n_optical_bands"] == 4
    assert img.shape[0] == meta["n_sar_bands"] + meta["n_optical_bands"]


def test_unlabeled_sar_sample_keeps_mask_none_even_if_optical_has_mask() -> None:
    opt_mask = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)
    sar_ds = _TupleDataset([_make_item("tile", channels=2, mask=None)])
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=4, mask=opt_mask, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds)
    _img, mask, meta = fused[0]

    assert mask is None
    assert meta["mask_path"] is None


def test_repr_reports_counts_and_policy() -> None:
    sar_ds = _TupleDataset(
        [_make_item("tile_1", channels=2), _make_item("tile_2", channels=2)]
    )
    opt_ds = _TupleDataset([_make_item("tile_1", channels=3, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    text = repr(fused)

    assert "n_sar=2" in text
    assert "n_optical=1" in text
    assert "n_paired=1" in text
    assert "policy='zeros'" in text


def test_infer_optical_bands_falls_back_to_one_when_optical_dataset_empty() -> None:
    sar_ds = _TupleDataset([_make_item("tile", channels=2)])
    opt_ds = _TupleDataset([], with_samples=True)

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    img, _mask, meta = fused[0]

    assert img.shape == (3, 2, 2)
    assert meta["n_optical_bands"] == 1
    assert torch.equal(img[2:], torch.zeros((1, 2, 2), dtype=torch.float32))


def test_sar_only_is_behaviorally_identical_to_zeros_alias() -> None:
    sar_ds = _TupleDataset([_make_item("missing_tile", channels=2)])
    opt_ds = _TupleDataset([_make_item("other_tile", channels=3, modality="optical")])

    zeros_ds = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    sar_only_ds = FusedDataset(sar_ds, opt_ds, optical_missing_policy="sar_only")

    zeros_img, zeros_mask, zeros_meta = zeros_ds[0]
    sar_only_img, sar_only_mask, sar_only_meta = sar_only_ds[0]

    assert torch.equal(zeros_img, sar_only_img)
    assert zeros_mask == sar_only_mask
    assert torch.equal(zeros_meta["valid_mask"], sar_only_meta["valid_mask"])
    assert zeros_meta["fused_optical_available"] == sar_only_meta["fused_optical_available"]
    assert zeros_meta["n_optical_bands"] == sar_only_meta["n_optical_bands"]
    assert zeros_meta["optical_img_path"] == sar_only_meta["optical_img_path"]
    assert zeros_ds.optical_missing_policy == "zeros"
    assert sar_only_ds.optical_missing_policy == "zeros"
