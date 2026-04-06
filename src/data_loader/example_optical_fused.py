"""
Quickstart examples for OpticalDataset and FusedDataset.

Patterns shown:
1. Optical-only training from raw Sentinel-2 roots
2. Strict paired SAR + optical fusion from Combined/manifest.csv
3. Legacy ad hoc fusion with missing-optical tolerance

Run with:
    python -m data_loader.example_optical_fused
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from data_loader import (
    FusedDataset,
    OpticalDataset,
    PatchDataset,
    SARDataset,
    compute_running_mean_std,
    default_collate,
    make_split,
    validate_sample_shapes,
)

SAR_IMG_ROOT = Path("datasets/FilteredSouthAsia/HandLabeled/S1Hand")
S2_IMG_ROOT = Path("datasets/FilteredSouthAsia/HandLabeled/S2Hand")
MASK_ROOT = Path("datasets/FilteredSouthAsia/HandLabeled/LabelHand")
SCL_ROOT = Path("datasets/FilteredSouthAsia/HandLabeled/S2SCL")
COMBINED_ROOT = Path("datasets/FilteredSouthAsia/Combined")

ids = ["tile_000123", "tile_000124", "tile_000125", "tile_000126"]
train_ids, val_ids = make_split(ids, val_frac=0.2, seed=1337)


def optical_only_example() -> None:
    print("\n-- Pattern 1: optical-only --")

    s2_raw = OpticalDataset(
        img_root=S2_IMG_ROOT,
        mask_root=MASK_ROOT,
        ids_or_paths=train_ids,
        mode="strong",
        s2_bands=[0, 1, 2, 3],
        cloud_mask_root=SCL_ROOT,
        cloud_mask_invalid_values=(3, 8, 9, 10, 11),
        normalize_cfg="none",
        validate=True,
    )
    mean, std = compute_running_mean_std(s2_raw, max_samples=512)
    print(f"S2 mean={mean} std={std}")

    s2_train = OpticalDataset(
        img_root=S2_IMG_ROOT,
        mask_root=MASK_ROOT,
        ids_or_paths=train_ids,
        mode="strong",
        s2_bands=[0, 1, 2, 3],
        cloud_mask_root=SCL_ROOT,
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
    )
    _s2_val = OpticalDataset(
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
    print(
        f"images={images.shape} masks={masks.shape} "
        f"valid_masks={valid_masks.shape} id={metas[0]['id']}"
    )


def strict_fused_example() -> None:
    print("\n-- Pattern 2: strict paired fusion from Combined/manifest.csv --")

    raw_fused = FusedDataset.from_combined_manifest(
        COMBINED_ROOT,
        mode="none",
        sar_dataset_kwargs={
            "normalize_cfg": "none",
            "log_transform": True,
        },
        optical_dataset_kwargs={
            "s2_bands": [0, 1, 2, 3],
            "normalize_cfg": "none",
        },
    )
    sar_mean, sar_std = compute_running_mean_std(raw_fused.sar_dataset, max_samples=256)
    s2_mean, s2_std = compute_running_mean_std(
        raw_fused.optical_dataset,
        max_samples=256,
    )

    fused_ds = FusedDataset.from_combined_manifest(
        COMBINED_ROOT,
        mode="none",
        sar_dataset_kwargs={
            "normalize_cfg": {"type": "zscore", "mean": sar_mean, "std": sar_std},
            "log_transform": True,
        },
        optical_dataset_kwargs={
            "s2_bands": [0, 1, 2, 3],
            "normalize_cfg": {"type": "zscore", "mean": s2_mean, "std": s2_std},
        },
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
    print(
        f"images={images.shape} masks={masks} valid_masks={valid_masks.shape} "
        f"sar_bands={metas[0]['n_sar_bands']} s2_bands={metas[0]['n_optical_bands']} "
        f"strict={metas[0]['strict_paired_mode']}"
    )


def fused_missing_optical_example() -> None:
    print("\n-- Pattern 3: legacy ad hoc fusion with missing-optical tolerance --")

    sar_train = SARDataset(
        img_root=SAR_IMG_ROOT,
        mask_root=MASK_ROOT,
        ids_or_paths=train_ids,
        mode="strong",
        normalize_cfg="none",
        log_transform=True,
    )

    partial_s2_ids = train_ids[:2]
    s2_train = OpticalDataset(
        img_root=S2_IMG_ROOT,
        mask_root=None,
        ids_or_paths=partial_s2_ids,
        mode="none",
        normalize_cfg="none",
    )

    fused_ds = FusedDataset(
        sar_dataset=sar_train,
        optical_dataset=s2_train,
        optical_missing_policy="zeros",
    )
    print(
        "This path is not strict paired multimodal training because membership "
        "comes from caller-supplied IDs rather than Combined/manifest.csv."
    )
    print(fused_ds)

    image, mask, meta = fused_ds[0]
    print(
        f"optical_available={meta['fused_optical_available']} "
        f"image_shape={image.shape} mask_is_none={mask is None}"
    )


if __name__ == "__main__":
    optical_only_example()
    strict_fused_example()
    fused_missing_optical_example()
