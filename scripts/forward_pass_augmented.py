from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import (  # noqa: E402
    PatchDataset,
    SARDataset,
    compute_running_mean_std,
    default_collate,
    list_ids_from_dir,
)
from src.models.unet.model import UNet  # noqa: E402


DEFAULT_MASK_SUFFIX_MAP = {
    "_S1Weak": "_S1OtsuLabelWeak",
    "_S1Hand": "_LabelHand",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize SAR data, build a patch-augmented dataset, and run a UNet forward pass."
    )
    parser.add_argument(
        "--img-root",
        type=Path,
        default=Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak"),
        help="Root directory for SAR images.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak"),
        help="Root directory for binary masks.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="weak",
        choices=["weak", "strong", "none"],
        help="Label mode for the dataset.",
    )
    parser.add_argument(
        "--max-ids",
        type=int,
        default=None,
        help="Optionally limit the number of IDs used for stats and dataset construction.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally limit samples for running mean/std computation.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size for PatchDataset augmentation.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Fractional overlap between patches (0-1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the forward pass.",
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        default=True,
        help="Apply log1p(abs(x)) before normalization.",
    )
    parser.add_argument(
        "--no-log-transform",
        action="store_false",
        dest="log_transform",
        help="Disable log transform.",
    )
    parser.add_argument(
        "--mask-suffix-map",
        type=str,
        default=None,
        help=(
            "Optional suffix map for resolving masks, formatted as "
            "'_S1Weak=_S1OtsuLabelWeak,_S1Hand=_LabelHand'. "
            "If omitted, repo defaults are used."
        ),
    )
    return parser.parse_args()


def parse_suffix_map(text: str | None) -> dict[str, str]:
    if text is None:
        return dict(DEFAULT_MASK_SUFFIX_MAP)
    text = text.strip()
    if not text:
        return {}
    pairs = [p.strip() for p in text.split(",") if p.strip()]
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid mask suffix map entry: {pair}")
        left, right = pair.split("=", 1)
        if not left or not right:
            raise ValueError(f"Invalid mask suffix map entry: {pair}")
        out[left] = right
    return out


def apply_suffix_map(sample_id: str, suffix_map: dict[str, str]) -> str:
    for img_suffix, mask_suffix in suffix_map.items():
        if sample_id.endswith(img_suffix):
            return f"{sample_id[:-len(img_suffix)]}{mask_suffix}"
    return sample_id


def resolve_raster_path(root: Path, name_or_id: str) -> Path:
    name = Path(name_or_id).name
    if name.lower().endswith((".tif", ".tiff")):
        return root / name
    tif = root / f"{name_or_id}.tif"
    tiff = root / f"{name_or_id}.tiff"
    if tif.exists():
        return tif
    if tiff.exists():
        return tiff
    return tif


def clean_nonfinite(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    metadata: dict,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image, mask, metadata


def main() -> None:
    args = get_args()

    if args.mode == "none":
        ids = list_ids_from_dir(args.img_root)
        report = {
            "images_total": len(ids),
            "paired_total": len(ids),
            "mode": "none",
        }
        suffix_map = {}
    else:
        suffix_map = parse_suffix_map(args.mask_suffix_map)
        image_ids = list_ids_from_dir(args.img_root)
        ids = []
        for sample_id in image_ids:
            mask_id = apply_suffix_map(sample_id, suffix_map)
            mask_path = resolve_raster_path(args.mask_root, mask_id)
            if mask_path.exists():
                ids.append(sample_id)
        report = {
            "images_total": len(image_ids),
            "paired_total": len(ids),
            "missing_in_masks": len(image_ids) - len(ids),
            "suffix_map": suffix_map,
        }

    if not ids:
        raise RuntimeError("No IDs found for the requested roots/mode.")
    if args.max_ids is not None:
        ids = ids[: args.max_ids]

    print(f"ID discovery report: {report}")
    print(f"Using {len(ids)} IDs")

    # 1) Raw dataset for stats
    raw_ds = SARDataset(
        img_root=args.img_root,
        mask_root=args.mask_root if args.mode != "none" else None,
        ids_or_paths=ids,
        mode=args.mode,
        normalize_cfg="none",
        log_transform=args.log_transform,
        transforms=clean_nonfinite,
        mask_id_suffix_map=suffix_map or None,
    )

    mean, std = compute_running_mean_std(
        raw_ds,
        max_samples=args.max_samples,
        require_no_transforms=False,
        include_log_transform_flag=args.log_transform,
    )

    print(f"mean: {mean}")
    print(f"std:  {std}")

    # 2) Normalized dataset
    norm_ds = SARDataset(
        img_root=args.img_root,
        mask_root=args.mask_root if args.mode != "none" else None,
        ids_or_paths=ids,
        mode=args.mode,
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
        log_transform=args.log_transform,
        transforms=clean_nonfinite,
        mask_id_suffix_map=suffix_map or None,
    )

    # 3) Patch augmentation
    patch_ds = PatchDataset(
        norm_ds,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )
    print(patch_ds)

    loader = DataLoader(
        patch_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=default_collate,
    )

    images, masks, metas = next(iter(loader))
    print(f"batch images: {tuple(images.shape)}")
    if masks is None:
        print("batch masks: None")
    else:
        print(f"batch masks:  {tuple(masks.shape)}")

    model = UNet(n_channels=images.shape[1], n_classes=1, bilinear=False)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    print(f"model outputs: {tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
