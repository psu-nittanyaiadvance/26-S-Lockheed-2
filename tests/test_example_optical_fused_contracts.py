from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import FusedDataset, default_collate  # noqa: E402

EXAMPLE_PATH = ROOT / "src" / "data_loader" / "example_optical_fused.py"


class _TupleDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]],
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
    mask: Optional[torch.Tensor],
    modality: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    img = torch.ones((channels, 2, 2), dtype=torch.float32)
    meta: Dict[str, Any] = {
        "id": sample_id,
        "img_path": f"{sample_id}.tif",
        "mask_path": f"{sample_id}_mask.tif" if mask is not None else None,
        "valid_mask": torch.ones((2, 2), dtype=torch.bool),
    }
    if modality is not None:
        meta["modality"] = modality
    return img, mask, meta


def test_example_loader_unpack_matches_default_collate_contract() -> None:
    batch = [
        _make_item("tile", channels=4, mask=torch.ones((1, 2, 2), dtype=torch.uint8))
    ]

    assert len(default_collate(batch)) == 4


def test_example_dataset_and_loader_contracts_are_distinct() -> None:
    batch = [
        _make_item("tile", channels=4, mask=torch.ones((1, 2, 2), dtype=torch.uint8))
    ]

    assert len(default_collate(batch)) == 4


def test_example_direct_fused_getitem_unpack_matches_dataset_contract() -> None:
    sar_ds = _TupleDataset(
        [_make_item("tile", channels=2, mask=torch.ones((1, 2, 2), dtype=torch.uint8))]
    )
    opt_ds = _TupleDataset([_make_item("tile", channels=3, mask=None, modality="optical")])
    fused = FusedDataset(sar_ds, opt_ds)

    assert len(fused[0]) == 3
