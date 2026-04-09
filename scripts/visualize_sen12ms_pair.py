from __future__ import annotations

"""Inspect strict paired SEN12MS samples after dataset preprocessing but before augmentation."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import FusedDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a strict paired SEN12MS SAR + optical sample after "
            "dataset preprocessing but before dataset augmentation. This does "
            "not show raw rasters."
        )
    )
    parser.add_argument(
        "combined_root",
        type=Path,
        help="Path to the Combined directory or its manifest.csv.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Manifest row index to visualize.",
    )
    parser.add_argument(
        "--mode",
        choices=["none", "weak", "strong"],
        default="none",
        help="Label loading mode. Use 'none' to inspect imagery without loading masks.",
    )
    parser.add_argument(
        "--s2-bands",
        type=str,
        default=None,
        help=(
            "Optional comma-separated zero-based Sentinel-2 band indices to load, "
            "for example '0,1,2,3'. Omit to load all bands; RGB interpretation "
            "then depends on the source band order."
        ),
    )
    parser.add_argument(
        "--optical-rgb-bands",
        type=str,
        default="2,1,0",
        help=(
            "Comma-separated zero-based optical tensor band indices to display as RGB. "
            "Defaults to '2,1,0', matching B4/B3/B2 only when --s2-bands is "
            "'0,1,2,3'."
        ),
    )
    parser.add_argument(
        "--sar-bands",
        type=str,
        default="0,1",
        help="Comma-separated zero-based SAR tensor band indices to display.",
    )
    parser.add_argument(
        "--sar-log-transform",
        action="store_true",
        help="Apply the existing SARDataset preprocessing log transform before visualization.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip joint manifest raster validation at dataset construction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the visualization image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the matplotlib window. By default, the script shows only when --output is omitted.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output figure DPI when --output is provided.",
    )
    return parser.parse_args()


def parse_band_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "all":
        return None
    bands = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not bands:
        return None
    if any(band < 0 for band in bands):
        raise ValueError(f"Band indices must be non-negative: {value!r}")
    return bands


def tensor_to_numpy_chw(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def stretch_for_display(arr: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    lo = float(np.nanpercentile(arr[finite], low_pct))
    hi = float(np.nanpercentile(arr[finite], high_pct))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def optical_display_image(optical: np.ndarray, rgb_bands: Sequence[int]) -> tuple[np.ndarray, str]:
    if optical.shape[0] >= 3 and len(rgb_bands) == 3 and max(rgb_bands) < optical.shape[0]:
        rgb = np.stack([stretch_for_display(optical[band]) for band in rgb_bands], axis=-1)
        return rgb, f"Optical RGB bands {list(rgb_bands)}"
    return stretch_for_display(optical[0]), "Optical band 0"


def selected_sar_bands(sar: np.ndarray, requested_bands: Optional[Sequence[int]]) -> list[int]:
    if requested_bands is None:
        requested_bands = [0, 1]
    return [band for band in requested_bands if band < sar.shape[0]]


def jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "paired_sample_id",
        "id",
        "sar_img_path",
        "optical_img_path",
        "label_path",
        "mask_path",
        "manifest_path",
        "manifest_row_index",
        "strict_paired_mode",
        "pairing_source",
        "fused_optical_available",
        "n_sar_bands",
        "n_optical_bands",
    ]
    return {key: metadata.get(key) for key in keep if key in metadata}


def plot_pair(
    *,
    sar: np.ndarray,
    optical: np.ndarray,
    mask: Optional[torch.Tensor],
    metadata: dict[str, Any],
    sar_bands: Sequence[int],
    optical_rgb_bands: Sequence[int],
    output: Optional[Path],
    show: bool,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    optical_image, optical_title = optical_display_image(optical, optical_rgb_bands)
    n_sar = len(sar_bands)
    has_mask = mask is not None
    n_cols = n_sar + 1 + (1 if has_mask else 0)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)
    axes_flat = axes[0]

    for axis, band in zip(axes_flat, sar_bands):
        axis.imshow(stretch_for_display(sar[band]), cmap="gray")
        axis.set_title(f"SAR band {band}")
        axis.axis("off")

    optical_axis = axes_flat[n_sar]
    if optical_image.ndim == 2:
        optical_axis.imshow(optical_image, cmap="gray")
    else:
        optical_axis.imshow(optical_image)
    optical_axis.set_title(optical_title)
    optical_axis.axis("off")

    if has_mask:
        mask_arr = tensor_to_numpy_chw(mask)[0]
        mask_axis = axes_flat[-1]
        mask_axis.imshow(mask_arr, cmap="gray")
        mask_axis.set_title("Label mask")
        mask_axis.axis("off")

    paired_id = metadata.get("paired_sample_id", metadata.get("id", "unknown"))
    fig.suptitle(f"SEN12MS paired sample: {paired_id}")
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved visualization: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    s2_bands = parse_band_list(args.s2_bands)
    sar_bands = parse_band_list(args.sar_bands)
    optical_rgb_bands = parse_band_list(args.optical_rgb_bands)
    if optical_rgb_bands is None or len(optical_rgb_bands) != 3:
        raise ValueError("--optical-rgb-bands must contain exactly three band indices")

    dataset = FusedDataset.from_combined_manifest(
        args.combined_root,
        mode=args.mode,
        sar_dataset_kwargs={
            "normalize_cfg": "none",
            "log_transform": args.sar_log_transform,
            "transforms": None,
        },
        optical_dataset_kwargs={
            "s2_bands": s2_bands,
            "normalize_cfg": "none",
            "transforms": None,
        },
        validate=not args.no_validate,
    )

    pair = dataset.get_paired_item(args.index, require_no_transforms=True)
    sar = tensor_to_numpy_chw(pair["sar_image"])
    optical = tensor_to_numpy_chw(pair["optical_image"])
    metadata = pair["metadata"]

    sar_display_bands = selected_sar_bands(sar, sar_bands)
    if not sar_display_bands:
        raise ValueError(f"No requested SAR bands are available; SAR shape is {sar.shape}")

    print("Paired SEN12MS metadata:")
    print(json.dumps(jsonable_metadata(metadata), indent=2))
    print(f"SAR tensor shape: {tuple(pair['sar_image'].shape)}")
    print(f"Optical tensor shape: {tuple(pair['optical_image'].shape)}")
    if pair["mask"] is not None:
        print(f"Mask tensor shape: {tuple(pair['mask'].shape)}")

    should_show = args.show or args.output is None
    plot_pair(
        sar=sar,
        optical=optical,
        mask=pair["mask"],
        metadata=metadata,
        sar_bands=sar_display_bands,
        optical_rgb_bands=optical_rgb_bands,
        output=args.output,
        show=should_show,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
