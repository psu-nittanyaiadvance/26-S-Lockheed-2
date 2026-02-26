from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import PatchDataset, default_collate  # noqa: E402
from data_loader.sar_dataset import SARDataset  # noqa: E402
import data_loader.patch_dataset as patch_module  # noqa: E402
from train import (  # noqa: E402
    _apply_ignore_index,
    _extract_ignore_mask,
    init_metric_state,
    update_metric_state,
)


class _DummyDataset(Dataset):
    def __init__(self) -> None:
        img0 = torch.zeros((1, 2, 2), dtype=torch.float32)
        mask0 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.uint8)
        meta0: Dict[str, Any] = {
            "id": "sample_0",
            "img_path": "dummy_0.tif",
            "mask_path": "dummy_0_mask.tif",
            "ignore_mask": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.bool),
        }

        img1 = torch.zeros((1, 2, 2), dtype=torch.float32)
        mask1 = torch.tensor([[[0, 0], [1, 0]]], dtype=torch.uint8)
        meta1: Dict[str, Any] = {
            "id": "sample_1",
            "img_path": "dummy_1.tif",
            "mask_path": "dummy_1_mask.tif",
        }

        self.samples: Tuple[
            Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]],
            Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]],
        ] = (
            (img0, mask0, meta0),
            (img1, mask1, meta1),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        return self.samples[idx]


class _DummySARDataset(SARDataset):
    def _load_image(self, img_path: str, sample_id: str) -> np.ndarray:
        return np.zeros((2, 2, 2), dtype=np.float32)

    def _load_mask(
        self, mask_path: str, expected_hw: Tuple[int, int], sample_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros(expected_hw, dtype=np.uint8)
        mask[0, 0] = 1
        ignore = np.zeros(expected_hw, dtype=bool)
        ignore[0, 0] = True
        return mask[None, :, :], ignore[None, :, :]


def test_ignore_mask_end_to_end_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(patch_module, "_tile_shape", lambda _: (2, 2))

    base_ds = _DummyDataset()
    patch_ds = PatchDataset(base_ds, patch_size=2, overlap=0.0)
    loader = DataLoader(patch_ds, batch_size=2, shuffle=False, collate_fn=default_collate)

    images, masks, _, metas = next(iter(loader))
    assert masks is not None

    ignore_mask = _extract_ignore_mask(metas, torch.device("cpu"), masks)
    assert ignore_mask is not None
    assert ignore_mask.shape == (2, 1, 2, 2)
    assert ignore_mask.dtype == torch.bool
    assert ignore_mask[0, 0, 0, 0].item() is True
    assert ignore_mask[1].sum().item() == 0

    logits = torch.ones_like(masks, dtype=torch.float32)
    state = init_metric_state(1)
    update_metric_state(state, logits, masks, 1, ignore_mask=ignore_mask)

    assert state["tp"] == 2.0
    assert state["fp"] == 5.0
    assert state["fn"] == 0.0
    assert state["tp"] + state["fp"] + state["fn"] == 7.0


def test_transform_contract_dropping_ignore_mask_raises(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    mask_root = tmp_path / "mask"
    img_root.mkdir()
    mask_root.mkdir()

    def bad_transform(img: torch.Tensor, mask: torch.Tensor, meta: Dict[str, Any]):
        meta.pop("ignore_mask", None)
        return img, mask

    ds = _DummySARDataset(
        img_root=img_root,
        mask_root=mask_root,
        ids_or_paths=["sample_0"],
        mode="strong",
        transforms=bad_transform,
    )

    with pytest.raises(ValueError, match="dropped metadata\\['ignore_mask'\\]"):
        _ = ds[0]


def test_patch_dataset_ignore_mask_alignment(monkeypatch) -> None:
    monkeypatch.setattr(patch_module, "_tile_shape", lambda _: (2, 2))

    class _MismatchDataset(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(
            self, idx: int
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
            img = torch.zeros((1, 2, 2), dtype=torch.float32)
            mask = torch.zeros((1, 2, 2), dtype=torch.uint8)
            ignore = torch.zeros((1, 1, 1), dtype=torch.bool)
            meta = {
                "id": "sample_mismatch",
                "img_path": "dummy.tif",
                "mask_path": "dummy_mask.tif",
                "ignore_mask": ignore,
            }
            return img, mask, meta

    patch_ds = PatchDataset(_MismatchDataset(), patch_size=2, overlap=0.0)
    with pytest.raises(ValueError, match="ignore_mask spatial shape mismatch"):
        _ = patch_ds[0]


def test_multiclass_ignore_index_matches_manual_masking() -> None:
    logits = torch.tensor(
        [[[[2.0, 0.5], [1.0, -1.0]], [[0.0, 1.5], [-0.5, 0.0]], [[-1.0, 0.0], [0.5, 1.0]]]]
    )
    targets = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    ignore_mask = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.bool)
    ignore_index = 255

    masked_targets = _apply_ignore_index(targets, ignore_mask, ignore_index)
    loss_with_ignore = torch.nn.functional.cross_entropy(
        logits, masked_targets, ignore_index=ignore_index
    )

    per_pixel = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    valid = ~ignore_mask.squeeze(1)
    manual_loss = per_pixel[valid].mean()

    assert loss_with_ignore.item() == pytest.approx(manual_loss.item())
