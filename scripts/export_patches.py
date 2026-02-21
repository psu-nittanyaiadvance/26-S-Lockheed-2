from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import (  # noqa: E402
    PatchDataset,
    SARDataset,
    compute_running_mean_std,
    list_ids_from_dir,
)


DEFAULT_MASK_SUFFIX_MAP = {
    "_S1Weak": "_S1OtsuLabelWeak",
    "_S1Hand": "_LabelHand",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export patch-augmented samples to disk."
    )
    parser.add_argument(
        "--img-root",
        type=Path,
        default=None,
        help="Root directory for SAR images.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
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
        "--out-dir",
        type=Path,
        default=Path("data/derived/patches/weak"),
        help="Output directory for patch files and metadata.",
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
        "--max-patches",
        type=int,
        default=None,
        help="Optionally limit number of patches exported.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume export by skipping existing patch files and appending metadata.",
    )
    parser.add_argument(
        "--reuse-stats",
        action="store_true",
        default=True,
        help="Reuse existing normalization stats if found in output dir.",
    )
    parser.add_argument(
        "--no-reuse-stats",
        action="store_false",
        dest="reuse_stats",
        help="Force recompute normalization stats.",
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
    args = parser.parse_args()

    if args.img_root is None:
        if args.mode == "strong":
            args.img_root = Path("datasets/FilteredSouthAsia/HandLabeled/S1Hand")
        elif args.mode == "weak":
            args.img_root = Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak")
    if args.mask_root is None:
        if args.mode == "strong":
            args.mask_root = Path("datasets/FilteredSouthAsia/HandLabeled/LabelHand")
        elif args.mode == "weak":
            args.mask_root = Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak")

    return args


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
    suffix_map = parse_suffix_map(args.mask_suffix_map)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "none":
        ids = list_ids_from_dir(args.img_root)
        report = {
            "images_total": len(ids),
            "paired_total": len(ids),
            "mode": "none",
        }
        suffix_map = {}
    else:
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

    stats_path = out_dir / "norm_stats.npz"
    if args.reuse_stats and stats_path.exists():
        stats = np.load(stats_path)
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)
        print("Loaded normalization stats from disk.")
    else:
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
        np.savez(stats_path, mean=mean, std=std)

    print(f"mean: {mean}")
    print(f"std:  {std}")

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

    patch_ds = PatchDataset(norm_ds, patch_size=args.patch_size, overlap=args.overlap)

    data_dir = out_dir / "patches"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"

    fields = [
        "id",
        "base_id",
        "patch_row",
        "patch_col",
        "patch_y0",
        "patch_x0",
        "patch_size",
        "img_path",
        "mask_path",
        "mode",
    ]

    n_total = len(patch_ds)
    n_target = n_total if args.max_patches is None else min(n_total, args.max_patches)
    existing: set[str] = set()
    if args.resume and data_dir.exists():
        existing = {p.stem for p in data_dir.glob("*.npz")}
        print(f"Resume mode: found {len(existing)} existing patch files.")
    print(f"Exporting up to {n_target} patches to {out_dir}")

    t0 = time.time()
    write_header = not (args.resume and meta_path.exists())
    with meta_path.open("a" if not write_header else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        exported = 0
        for idx in range(n_total):
            if exported >= n_target:
                break

            patch_id = None
            meta = None

            if hasattr(patch_ds, "_index") and hasattr(norm_ds, "samples"):
                pi = patch_ds._index[idx]
                base_sample = norm_ds.samples[pi.base_idx]
                base_id = base_sample["id"]
                patch_id = f"{base_id}_r{pi.row}_c{pi.col}"
                if args.resume and patch_id in existing:
                    continue
                meta = {
                    "id": patch_id,
                    "base_id": base_id,
                    "patch_row": pi.row,
                    "patch_col": pi.col,
                    "patch_y0": pi.y0,
                    "patch_x0": pi.x0,
                    "patch_size": patch_ds.patch_size,
                    "img_path": base_sample["img_path"],
                    "mask_path": base_sample["mask_path"],
                }

            img, mask, meta = patch_ds[idx] if meta is None else (patch_ds[idx][0], patch_ds[idx][1], meta)
            patch_id = meta["id"]

            out_file = data_dir / f"{patch_id}.npz"
            if args.resume and out_file.exists():
                continue

            img_np = img.detach().cpu().numpy().astype(np.float32, copy=False)
            if mask is None:
                np.savez_compressed(out_file, image=img_np)
            else:
                mask_np = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
                np.savez_compressed(out_file, image=img_np, mask=mask_np)

            row = {
                "id": patch_id,
                "base_id": meta.get("base_id"),
                "patch_row": meta.get("patch_row"),
                "patch_col": meta.get("patch_col"),
                "patch_y0": meta.get("patch_y0"),
                "patch_x0": meta.get("patch_x0"),
                "patch_size": meta.get("patch_size"),
                "img_path": meta.get("img_path"),
                "mask_path": meta.get("mask_path"),
                "mode": args.mode,
            }
            writer.writerow(row)
            exported += 1

            if exported % 200 == 0 or exported == n_target:
                elapsed = time.time() - t0
                print(f"[{exported}/{n_target}] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
