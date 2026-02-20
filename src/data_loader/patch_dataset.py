"""
patch_dataset.py  –  Overlapping-patch tiling for small SAR datasets.

Motivation
----------
138 hand-labeled tiles are far too few to train a U-Net from scratch without
overfitting.  By slicing each tile into overlapping 256×256 patches we can
expand the effective training set dramatically.

With 20 % overlap (stride = 205 px) a single 512×512 tile yields a 2×2 grid
of non-overlapping patches, but a slightly larger tile (e.g. 900×900) yields
roughly 25 patches.  For typical Sentinel-1 GRD tiles (~1000–2000 px), you
should expect 20–80 patches per image, turning 138 images into 2,760–11,040
training patches – a much more viable regime.

Usage
-----
    from dataloader.sar_dataset import SARDataset
    from dataloader.patch_dataset import PatchDataset

    base_ds = SARDataset(
        img_root=img_root,
        mask_root=mask_root,
        ids_or_paths=train_ids,
        mode="strong",
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
        log_transform=True,
    )

    patch_ds = PatchDataset(base_ds, patch_size=256, overlap=0.2)
    print(f"Base samples : {len(base_ds)}")
    print(f"Patch samples: {len(patch_ds)}")   # e.g. 138 → ~5000+

    loader = DataLoader(patch_ds, batch_size=8, shuffle=True,
                        num_workers=4, collate_fn=default_collate)

Design notes
------------
* Patches are enumerated lazily at init from image metadata (no pixels read).
* The final column / row of patches is *right-/bottom-aligned* so the full
  image area is always covered even when width/height is not divisible by the
  stride.
* Works with mode="none" (no mask) and mode="weak" / "strong" (binary mask).
* The metadata dict is forwarded from the base dataset and extended with
  patch-specific keys: patch_row, patch_col, patch_x0, patch_y0.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# ── helpers ──────────────────────────────────────────────────────────────────

def _grid_starts(length: int, patch: int, stride: int) -> List[int]:
    """
    Return the list of top-left positions that tile [0, length) with the
    given patch size and stride, guaranteeing full coverage.

    The last position is clamped so its patch doesn't exceed `length`.
    """
    if patch >= length:
        return [0]
    starts: List[int] = list(range(0, length - patch, stride))
    # always include the right / bottom edge
    last = length - patch
    if not starts or starts[-1] < last:
        starts.append(last)
    return starts


def _tile_shape(img_path: str) -> Tuple[int, int]:
    """Read (H, W) from a rasterio-readable file without loading pixels."""
    with rasterio.open(img_path) as src:
        return src.height, src.width


# ── PatchIndex ───────────────────────────────────────────────────────────────

class _PatchIndex:
    """Lightweight index: for each patch stores (base_idx, row, col, y0, x0)."""

    __slots__ = ("base_idx", "row", "col", "y0", "x0")

    def __init__(self, base_idx: int, row: int, col: int, y0: int, x0: int) -> None:
        self.base_idx = base_idx
        self.row = row
        self.col = col
        self.y0 = y0
        self.x0 = x0


# ── PatchDataset ─────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Wraps a SARDataset (or any dataset returning (image, mask, meta)) and
    exposes every overlapping patch as an independent sample.

    Parameters
    ----------
    base_dataset:
        The underlying SARDataset instance.
    patch_size:
        Spatial size of each square patch (pixels).  Default: 256.
    overlap:
        Fractional overlap between adjacent patches, in [0, 1).  Default: 0.2
        (20 %).  A value of 0 means non-overlapping.
    skip_mostly_nodata:
        If True, patches where > `nodata_threshold` fraction of the *mask*
        pixels are zero (background) are silently dropped at init time.
        Useful for coastal / edge tiles with large water areas.  Only
        applies when the base dataset has a mask (mode != "none").
        Default: False.
    nodata_threshold:
        Fraction of mask pixels that must be non-zero to keep a patch when
        `skip_mostly_nodata=True`.  Default: 0.0 (keep all non-empty patches).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        patch_size: int = 256,
        overlap: float = 0.2,
        skip_mostly_nodata: bool = False,
        nodata_threshold: float = 0.0,
    ) -> None:
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1); got {overlap}")
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1; got {patch_size}")

        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.overlap = overlap
        self.skip_mostly_nodata = skip_mostly_nodata
        self.nodata_threshold = nodata_threshold

        stride = max(1, int(math.ceil(patch_size * (1.0 - overlap))))
        self.stride = stride

        self._index: List[_PatchIndex] = self._build_index()

    # ── index building ───────────────────────────────────────────────────────

    def _build_index(self) -> List[_PatchIndex]:
        index: List[_PatchIndex] = []
        base = self.base_dataset

        for base_idx in range(len(base)):  # type: ignore[arg-type]
            try:
                _, _, meta = base[base_idx]
                img_path = meta.get("img_path", "")
                if not img_path:
                    warnings.warn(
                        f"Sample at index {base_idx} has no 'img_path' in metadata; "
                        "skipping."
                    )
                    continue
                h, w = _tile_shape(img_path)
            except Exception as exc:
                warnings.warn(
                    f"Could not read shape for base index {base_idx}: {exc}; skipping."
                )
                continue

            y_starts = _grid_starts(h, self.patch_size, self.stride)
            x_starts = _grid_starts(w, self.patch_size, self.stride)

            for row, y0 in enumerate(y_starts):
                for col, x0 in enumerate(x_starts):
                    index.append(_PatchIndex(base_idx, row, col, y0, x0))

        if not index:
            raise ValueError(
                "PatchDataset index is empty – no patches were generated. "
                "Check that the base dataset is non-empty and tiles are "
                "at least patch_size pixels on each side."
            )
        return index

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        pi = self._index[idx]
        img_tensor, mask_tensor, meta = self.base_dataset[pi.base_idx]

        ps = self.patch_size
        y0, x0 = pi.y0, pi.x0

        # Crop image: [C, H, W] → [C, ps, ps]
        img_patch = img_tensor[:, y0 : y0 + ps, x0 : x0 + ps]

        # Crop mask (if present): [1, H, W] → [1, ps, ps]
        mask_patch: Optional[torch.Tensor] = None
        if mask_tensor is not None:
            mask_patch = mask_tensor[:, y0 : y0 + ps, x0 : x0 + ps]

        # Extend metadata
        patch_meta: Dict[str, Any] = dict(meta)
        base_id = meta.get("id", f"base_{pi.base_idx}")
        patch_meta["id"] = f"{base_id}_r{pi.row}_c{pi.col}"
        patch_meta["base_id"] = base_id
        patch_meta["patch_row"] = pi.row
        patch_meta["patch_col"] = pi.col
        patch_meta["patch_y0"] = y0
        patch_meta["patch_x0"] = x0
        patch_meta["patch_size"] = ps
        if "ignore_mask" in meta:
            ignore_mask = meta["ignore_mask"]
            if ignore_mask is not None:
                if not isinstance(ignore_mask, torch.Tensor):
                    ignore_mask = torch.as_tensor(ignore_mask)
                if ignore_mask.ndim == 3:
                    patch_meta["ignore_mask"] = ignore_mask[:, y0 : y0 + ps, x0 : x0 + ps]
                elif ignore_mask.ndim == 2:
                    patch_meta["ignore_mask"] = ignore_mask[y0 : y0 + ps, x0 : x0 + ps]
                else:
                    base_id = meta.get("id", f"base_{pi.base_idx}")
                    raise ValueError(
                        "ignore_mask must have shape [1,H,W] or [H,W]; "
                        f"got {tuple(ignore_mask.shape)} (id={base_id} "
                        f"patch_y0={y0} patch_x0={x0})"
                    )
        if mask_patch is not None and "ignore_mask" in patch_meta:
            ignore_mask = patch_meta["ignore_mask"]
            if isinstance(ignore_mask, torch.Tensor):
                if ignore_mask.ndim == 3:
                    ignore_hw = tuple(ignore_mask.shape[1:])
                elif ignore_mask.ndim == 2:
                    ignore_hw = tuple(ignore_mask.shape)
                else:
                    base_id = meta.get("id", f"base_{pi.base_idx}")
                    raise ValueError(
                        "ignore_mask must have shape [1,H,W] or [H,W]; "
                        f"got {tuple(ignore_mask.shape)} (id={base_id} "
                        f"patch_y0={y0} patch_x0={x0})"
                    )
            else:
                ignore_hw = tuple(torch.as_tensor(ignore_mask).shape[-2:])
            mask_hw = tuple(mask_patch.shape[-2:])
            if ignore_hw != mask_hw:
                base_id = meta.get("id", f"base_{pi.base_idx}")
                raise ValueError(
                    "ignore_mask spatial shape mismatch: "
                    f"got {ignore_hw}, expected {mask_hw} (id={base_id} "
                    f"patch_y0={y0} patch_x0={x0})"
                )

        return img_patch, mask_patch, patch_meta

    # ── utility ──────────────────────────────────────────────────────────────

    def patches_per_image_stats(self) -> Dict[str, float]:
        """Return mean / min / max patches per base image (for diagnostics)."""
        from collections import Counter

        counts = Counter(pi.base_idx for pi in self._index)
        vals = list(counts.values())
        n = len(vals)
        return {
            "n_base_images": n,
            "n_patches_total": len(self._index),
            "mean_patches_per_image": sum(vals) / n if n else 0.0,
            "min_patches_per_image": min(vals) if vals else 0,
            "max_patches_per_image": max(vals) if vals else 0,
        }

    def __repr__(self) -> str:
        stats = self.patches_per_image_stats()
        return (
            f"PatchDataset("
            f"patch_size={self.patch_size}, "
            f"overlap={self.overlap:.0%}, "
            f"stride={self.stride}, "
            f"n_base={stats['n_base_images']}, "
            f"n_patches={stats['n_patches_total']}, "
            f"mean_per_image={stats['mean_patches_per_image']:.1f})"
        )
