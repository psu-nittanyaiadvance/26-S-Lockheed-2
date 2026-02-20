from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import PatchDataset, default_collate  # noqa: E402
import data_loader.patch_dataset as patch_module  # noqa: E402
from train import _extract_ignore_mask, init_metric_state, update_metric_state  # noqa: E402


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


def test_ignore_mask_end_to_end_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(patch_module, "_tile_shape", lambda _: (2, 2))

    base_ds = _DummyDataset()
    patch_ds = PatchDataset(base_ds, patch_size=2, overlap=0.0)
    loader = DataLoader(patch_ds, batch_size=2, shuffle=False, collate_fn=default_collate)

    images, masks, metas = next(iter(loader))
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
