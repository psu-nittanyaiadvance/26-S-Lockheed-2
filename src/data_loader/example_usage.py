"""
README-style quickstart:

1) Set IMG_ROOT, WEAK_MASK_ROOT, STRONG_MASK_ROOT to your dataset folders.
2) Provide IDs like "tile_000123" or a list of image paths.
3) Run this script to verify the loader and batch shapes.

Example IDs:
- IDs: ["tile_000123", "tile_000124", ...]
- Paths: ["D:/data/s1_tiles/tile_000123.tif", ...]
"""

from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from data_loader.sar_dataset import SARDataset
from data_loader.splits import make_split
from data_loader.collate import default_collate
from data_loader.stats import compute_running_mean_std


def main() -> None:
    img_root = Path("path/to/s1_tiles")
    weak_mask_root = Path("path/to/weak_masks")
    strong_mask_root = Path("path/to/hand_masks")

    ids = ["tile_000123", "tile_000124", "tile_000125", "tile_000126"]
    train_ids, val_ids = make_split(ids, val_frac=0.2, seed=1337)

    weak_train = SARDataset(
        img_root=img_root,
        mask_root=weak_mask_root,
        ids_or_paths=train_ids,
        mode="weak",
        normalize_cfg="none",
        log_transform=True,
        validate=True,
    )

    mean, std = compute_running_mean_std(weak_train)
    weak_train_norm = SARDataset(
        img_root=img_root,
        mask_root=weak_mask_root,
        ids_or_paths=train_ids,
        mode="weak",
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
        log_transform=True,
        validate=False,
    )

    weak_val = SARDataset(
        img_root=img_root,
        mask_root=weak_mask_root,
        ids_or_paths=val_ids,
        mode="weak",
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
        log_transform=True,
        validate=False,
    )

    weak_loader = DataLoader(
        weak_train_norm,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=default_collate,
    )

    strong_train = SARDataset(
        img_root=img_root,
        mask_root=strong_mask_root,
        ids_or_paths=train_ids,
        mode="strong",
        normalize_cfg="none",
        log_transform=True,
        validate=True,
    )

    strong_loader = DataLoader(
        strong_train,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=default_collate,
    )

    infer_ds = SARDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=val_ids,
        mode="none",
        normalize_cfg="none",
        log_transform=True,
        validate=False,
    )

    infer_loader = DataLoader(
        infer_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=default_collate,
    )

    images, masks, metas = next(iter(weak_loader))
    print(images.shape, masks.shape, metas[0])


if __name__ == "__main__":
    main()
