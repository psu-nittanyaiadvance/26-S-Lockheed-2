"""
example_optical_fused.py  –  Quickstart for OpticalDataset and FusedDataset.

Three patterns are shown:
  1. Optical-only training (S2 bands, cloud masking, zscore normalisation)
  2. SAR + optical fusion  (FusedDataset, concatenated channels)
  3. Fusion with missing-optical tolerance ("zeros" policy)

Set the path variables at the top to match your directory layout, then run:
    python example_optical_fused.py
"""

from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from dataloader import (
    SARDataset,
    OpticalDataset,
    FusedDataset,
    PatchDataset,
    make_split,
    compute_running_mean_std,
    default_collate,
    validate_sample_shapes,
)

# ── Paths – adjust to your layout ────────────────────────────────────────────

SAR_IMG_ROOT  = Path("datasets/FilteredSouthAsia/HandLabeled/S1Hand")
S2_IMG_ROOT   = Path("datasets/FilteredSouthAsia/HandLabeled/S2Hand")  # S2 tiles
MASK_ROOT     = Path("datasets/FilteredSouthAsia/HandLabeled/LabelHand")
SCL_ROOT      = Path("datasets/FilteredSouthAsia/HandLabeled/S2SCL")   # optional SCL

# ── IDs shared between SAR and S2 ────────────────────────────────────────────

ids = ["tile_000123", "tile_000124", "tile_000125", "tile_000126"]
train_ids, val_ids = make_split(ids, val_frac=0.2, seed=1337)


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern 1 – Optical-only
# ═══════════════════════════════════════════════════════════════════════════════

def optical_only_example() -> None:
    print("\n── Pattern 1: Optical-only ──")

    # Step 1: compute per-band stats on raw reflectance (normalize_cfg="none")
    s2_raw = OpticalDataset(
        img_root=S2_IMG_ROOT,
        mask_root=MASK_ROOT,
        ids_or_paths=train_ids,
        mode="strong",
        # Load B2, B3, B4, B8 (RGB + NIR) as zero-based indices
        s2_bands=[0, 1, 2, 3],
        cloud_mask_root=SCL_ROOT,           # optional; pass None to skip
        cloud_mask_invalid_values=(3, 8, 9, 10, 11),
        normalize_cfg="none",
        validate=True,
    )
    mean, std = compute_running_mean_std(s2_raw, max_samples=512)
    print(f"S2 per-band mean: {mean}  std: {std}")

    # Step 2: build normalised dataset for training
    s2_train = OpticalDataset(
        img_root=S2_IMG_ROOT,
        mask_root=MASK_ROOT,
        ids_or_paths=train_ids,
        mode="strong",
        s2_bands=[0, 1, 2, 3],
        cloud_mask_root=SCL_ROOT,
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
    )
    s2_val = OpticalDataset(
        img_root=S2_IMG_ROOT,
        mask_root=MASK_ROOT,
        ids_or_paths=val_ids,
        mode="strong",
        s2_bands=[0, 1, 2, 3],
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
    )

    validate_sample_shapes(s2_train, n=8)

    patch_ds = PatchDataset(s2_train, patch_size=256, overlap=0.2)
    loader = DataLoader(
        patch_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=default_collate,
    )
    images, masks, valid_masks, metas = next(iter(loader))
    # images: [B, 4, 256, 256] – four S2 bands
    print(f"images: {images.shape}  masks: {masks.shape}  "
          f"valid_masks: {valid_masks.shape}  id: {metas[0]['id']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern 2 – SAR + Optical fusion (fully paired dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def fused_example() -> None:
    print("\n── Pattern 2: SAR + Optical fusion ──")

    # --- SAR ---
    sar_raw = SARDataset(
        img_root=SAR_IMG_ROOT, mask_root=MASK_ROOT,
        ids_or_paths=train_ids, mode="strong",
        normalize_cfg="none", log_transform=True,
    )
    sar_mean, sar_std = compute_running_mean_std(sar_raw, max_samples=256)

    sar_train = SARDataset(
        img_root=SAR_IMG_ROOT, mask_root=MASK_ROOT,
        ids_or_paths=train_ids, mode="strong",
        normalize_cfg={"type": "zscore", "mean": sar_mean, "std": sar_std},
        log_transform=True,
    )

    # --- S2 (no mask needed – mask comes from SAR dataset) ---
    s2_raw = OpticalDataset(
        img_root=S2_IMG_ROOT, mask_root=None,
        ids_or_paths=train_ids, mode="none",
        s2_bands=[0, 1, 2, 3],
        normalize_cfg="none",
    )
    s2_mean, s2_std = compute_running_mean_std(s2_raw, max_samples=256)

    s2_train = OpticalDataset(
        img_root=S2_IMG_ROOT, mask_root=None,
        ids_or_paths=train_ids, mode="none",
        s2_bands=[0, 1, 2, 3],
        normalize_cfg={"type": "zscore", "mean": s2_mean, "std": s2_std},
    )

    # --- Fuse ---
    fused_ds = FusedDataset(
        sar_dataset=sar_train,
        optical_dataset=s2_train,
        optical_missing_policy="raise",  # all IDs must be paired
    )
    print(fused_ds)

    validate_sample_shapes(fused_ds, n=4)

    patch_ds = PatchDataset(fused_ds, patch_size=256, overlap=0.2)
    loader = DataLoader(
        patch_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=default_collate,
    )
    images, masks, valid_masks, metas = next(iter(loader))
    # images: [B, C_sar+C_s2, 256, 256]
    print(f"fused images: {images.shape}  (SAR bands: {metas[0]['n_sar_bands']}, "
          f"S2 bands: {metas[0]['n_optical_bands']})")
    print(f"masks: {masks.shape}  valid_masks: {valid_masks.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern 3 – Fusion tolerating missing optical tiles
# ═══════════════════════════════════════════════════════════════════════════════

def fused_missing_optical_example() -> None:
    print("\n── Pattern 3: Fusion with missing-optical tolerance ──")

    sar_train = SARDataset(
        img_root=SAR_IMG_ROOT, mask_root=MASK_ROOT,
        ids_or_paths=train_ids, mode="strong",
        normalize_cfg="none", log_transform=True,
    )

    # Suppose only a subset of tiles have S2 coverage
    partial_s2_ids = train_ids[:2]  # only first two tiles have S2
    s2_train = OpticalDataset(
        img_root=S2_IMG_ROOT, mask_root=None,
        ids_or_paths=partial_s2_ids, mode="none",
        normalize_cfg="none",
    )

    fused_ds = FusedDataset(
        sar_dataset=sar_train,
        optical_dataset=s2_train,
        optical_missing_policy="zeros",  # substitute zeros for missing S2
    )
    print(fused_ds)

    images, masks, valid_masks, metas = fused_ds[0]
    print(f"optical available: {metas['fused_optical_available']}")
    print(f"images: {images.shape}  valid coverage: "
          f"{valid_masks.float().mean():.1%}")

    # --- Loss masking pattern (copy into your training loop) ---
    # import torch
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    # pred = model(images.unsqueeze(0))           # [1, 1, H, W]
    # loss_px = criterion(pred, masks.float().unsqueeze(0))
    # vm = valid_masks.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # loss = (loss_px * vm).sum() / vm.sum().clamp(min=1)


if __name__ == "__main__":
    optical_only_example()
    fused_example()
    fused_missing_optical_example()
