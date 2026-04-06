from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from torch.utils.data import Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_loader.patch_dataset as patch_module  # noqa: E402
from data_loader import PatchDataset  # noqa: E402
from train import active_dict_collate, focal_tversky_loss  # noqa: E402


def _manual_focal_tversky(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    valid_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if valid_mask is None:
        valid = torch.ones_like(targets, dtype=probs.dtype)
    else:
        valid = valid_mask.unsqueeze(1).to(dtype=probs.dtype)

    probs = probs.view(probs.shape[0], probs.shape[1], -1)
    targets = targets.view(targets.shape[0], targets.shape[1], -1)
    valid = valid.view(valid.shape[0], 1, -1)

    tp = (probs * targets * valid).sum(dim=2)
    fp = ((1 - targets) * probs * valid).sum(dim=2)
    fn = (targets * (1 - probs) * valid).sum(dim=2)
    ti = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    base = (1.0 - ti).clamp(min=epsilon, max=1.0)
    return base.pow(gamma).mean()


class _TinyBaseDataset(Dataset):
    def __init__(self, count: int = 4) -> None:
        self.samples = []
        for i in range(count):
            tile_id = f"tile_{i}"
            image = torch.full((2, 4, 4), float(i), dtype=torch.float32)
            mask = torch.full((1, 4, 4), i % 2, dtype=torch.uint8)
            meta: Dict[str, Any] = {
                "id": tile_id,
                "img_path": f"{tile_id}.tif",
                "mask_path": f"{tile_id}_mask.tif",
                "valid_mask": torch.ones((4, 4), dtype=torch.bool),
            }
            self.samples.append((image, mask, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        return self.samples[idx]


def test_focal_tversky_matches_standard_formula_and_valid_mask() -> None:
    logits = torch.tensor([[[[2.0, -2.0], [0.0, 0.0]]]], dtype=torch.float32)
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    valid_mask = torch.tensor([[[1, 1], [1, 0]]], dtype=torch.bool)

    loss = focal_tversky_loss(
        logits,
        targets,
        alpha=0.4,
        beta=0.6,
        gamma=4.0 / 3.0,
        valid_mask=valid_mask,
    )
    expected = _manual_focal_tversky(
        logits,
        targets,
        alpha=0.4,
        beta=0.6,
        gamma=4.0 / 3.0,
        valid_mask=valid_mask,
    )
    unmasked = focal_tversky_loss(
        logits,
        targets,
        alpha=0.4,
        beta=0.6,
        gamma=4.0 / 3.0,
        valid_mask=torch.ones_like(valid_mask),
    )

    assert loss.item() == pytest.approx(expected.item())
    assert loss.item() < unmasked.item()


def test_active_dict_collate_batches_and_crops_valid_masks() -> None:
    batch = [
        (
            torch.zeros((2, 2, 2), dtype=torch.float32),
            torch.tensor([[[1, 0], [0, 1]]], dtype=torch.uint8),
            {
                "id": "tile_a_r0_c0",
                "base_id": "tile_a",
                "patch_y0": 1,
                "patch_x0": 1,
                "patch_size": 2,
                "valid_mask": torch.tensor(
                    [[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.bool
                ),
            },
        ),
        (
            torch.zeros((2, 2, 2), dtype=torch.float32),
            torch.tensor([[[0, 0], [1, 1]]], dtype=torch.uint8),
            {
                "id": "tile_b_r0_c0",
                "base_id": "tile_b",
                "patch_y0": 0,
                "patch_x0": 0,
                "patch_size": 2,
                "valid_mask": torch.tensor([[1, 1], [0, 1]], dtype=torch.bool),
            },
        ),
    ]

    collated = active_dict_collate(batch)

    assert tuple(collated["image"].shape) == (2, 2, 2, 2)
    assert tuple(collated["mask"].shape) == (2, 1, 2, 2)
    assert tuple(collated["valid_mask"].shape) == (2, 2, 2)
    assert collated["valid_mask"].dtype == torch.bool
    assert collated["valid_mask"][0].tolist() == [[1, 0], [1, 0]]
    assert collated["meta"][0]["base_id"] == "tile_a"


def test_split_before_patching_keeps_parent_tiles_disjoint_and_patch_provenance() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(patch_module, "_tile_shape", lambda _: (4, 4))
    try:
        base_dataset = _TinyBaseDataset(count=4)
        generator = torch.Generator().manual_seed(0)
        train_base, val_base = random_split(base_dataset, [3, 1], generator=generator)

        train_patch_ds = PatchDataset(train_base, patch_size=2, overlap=0.0)
        val_patch_ds = PatchDataset(val_base, patch_size=2, overlap=0.0)

        train_base_ids = {train_patch_ds[i][2]["base_id"] for i in range(len(train_patch_ds))}
        val_base_ids = {val_patch_ds[i][2]["base_id"] for i in range(len(val_patch_ds))}
        _, _, meta = train_patch_ds[0]

        assert train_base_ids.isdisjoint(val_base_ids)
        assert meta["id"].startswith(f"{meta['base_id']}_r")
        assert tuple(meta["valid_mask"].shape) == (2, 2)
        assert {"patch_row", "patch_col", "patch_y0", "patch_x0", "patch_size"} <= set(meta)
    finally:
        monkeypatch.undo()
