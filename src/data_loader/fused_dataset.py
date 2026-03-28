"""
fused_dataset.py  –  Late-pixel fusion of SAR (S1) and Optical (S2) datasets.

Overview
--------
FusedDataset pairs a SARDataset and an OpticalDataset by sample ID, then
concatenates their image tensors along the channel dimension:

    [C_sar + C_s2, H, W]

The combined valid_mask is the logical AND of both modality masks, so any
pixel invalid in *either* sensor is excluded from the loss.

Spatial alignment
-----------------
SAR and optical tiles must already be co-registered and share the same
(H, W) grid.  FusedDataset does **not** reproject or resample – if the grids
differ a clear error is raised at __getitem__ time.

Optical-missing policy
----------------------
In operational settings the optical tile for a given date may be missing or
entirely cloud-covered.  Three policies handle this gracefully:

    "zeros"  – substitute a zero tensor for the optical bands and mark all
               optical pixels as invalid.  The model sees the SAR bands as
               normal, and the fused valid_mask correctly masks out the zeros.
    "sar_only" – same as "zeros" but sets a metadata flag
               ``fused_optical_available=False`` so the model / loss can
               choose to ignore optical channels entirely for that sample.
    "raise"  – raise RuntimeError for any missing optical sample (safe default
               for fully paired datasets).

Usage
-----
    from dataloader.fused_dataset import FusedDataset

    sar_ds = SARDataset(img_root=..., mask_root=..., ids_or_paths=train_ids,
                        mode="strong", normalize_cfg=sar_norm, log_transform=True)
    s2_ds  = OpticalDataset(img_root=..., mask_root=..., ids_or_paths=train_ids,
                            mode="none",   normalize_cfg=s2_norm)

    fused = FusedDataset(sar_ds, s2_ds, optical_missing_policy="zeros")
    # Each sample: image [C_sar+C_s2, H, W], mask [1,H,W], metadata dict
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FusedDataset(Dataset):
    """
    Combines a SARDataset and an OpticalDataset, aligning samples by ID.

    The combined image tensor has shape ``[C_sar + C_s2, H, W]`` with SAR
    bands first.  The mask comes from ``sar_dataset`` (or ``optical_dataset``
    if the SAR dataset has mode="none" and optical has a mask).

    Parameters
    ----------
    sar_dataset : SARDataset
        Sentinel-1 dataset instance.  Must be initialised before being
        passed here (normalisation, log-transform, etc. already configured).
    optical_dataset : OpticalDataset
        Sentinel-2 dataset instance.
    optical_missing_policy : {"zeros", "sar_only", "raise"}
        How to handle SAR IDs that have no paired optical tile.
    require_spatial_match : bool
        If True (default), raise ValueError if SAR and optical spatial dims
        differ at __getitem__ time.  Set False only if you handle resampling
        in transforms.
    """

    def __init__(
        self,
        sar_dataset: Dataset,
        optical_dataset: Dataset,
        optical_missing_policy: str = "raise",
        require_spatial_match: bool = True,
    ) -> None:
        if optical_missing_policy not in {"zeros", "sar_only", "raise"}:
            raise ValueError(
                "optical_missing_policy must be one of {'zeros', 'sar_only', 'raise'}; "
                f"got {optical_missing_policy!r}"
            )

        self.sar_dataset = sar_dataset
        self.optical_dataset = optical_dataset
        self.optical_missing_policy = optical_missing_policy
        self.require_spatial_match = require_spatial_match

        # Build a fast ID → optical-index lookup
        self._optical_id_to_idx: Dict[str, int] = {}
        for i in range(len(optical_dataset)):  # type: ignore[arg-type]
            # Access the samples list if it exists (SARDataset / OpticalDataset),
            # otherwise fall back to __getitem__ to extract the id from metadata.
            samples_attr = getattr(optical_dataset, "samples", None)
            if samples_attr is not None:
                oid = samples_attr[i]["id"]
            else:
                _, _, meta = optical_dataset[i]
                oid = meta.get("id", str(i))
            self._optical_id_to_idx[oid] = i

        # Validate that all SAR IDs have a paired optical tile (for "raise" policy)
        if optical_missing_policy == "raise":
            sar_samples = getattr(sar_dataset, "samples", None)
            if sar_samples is not None:
                missing = [
                    s["id"] for s in sar_samples
                    if s["id"] not in self._optical_id_to_idx
                ]
                if missing:
                    raise ValueError(
                        f"optical_missing_policy='raise' but {len(missing)} SAR IDs "
                        f"have no paired optical tile. First few: {missing[:5]}"
                    )

    def __len__(self) -> int:
        return len(self.sar_dataset)  # type: ignore[arg-type]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        sar_img, mask_tensor, sar_meta = self.sar_dataset[idx]
        sample_id: str = sar_meta.get("id", str(idx))

        sar_valid: torch.Tensor = sar_meta.get(
            "valid_mask", torch.ones(sar_img.shape[-2:], dtype=torch.bool)
        )

        opt_idx = self._optical_id_to_idx.get(sample_id)

        # ── optical tile unavailable ─────────────────────────────────────────
        if opt_idx is None:
            if self.optical_missing_policy == "raise":
                raise RuntimeError(
                    f"No paired optical tile for SAR id='{sample_id}'"
                )
            # Synthesize a zero optical tensor of matching spatial size
            h, w = sar_img.shape[-2], sar_img.shape[-1]
            # We don't know C_s2 at construction time; use a single zero band
            # if the optical dataset is empty, else infer from first sample.
            n_opt_bands = self._infer_optical_bands()
            opt_img = torch.zeros(n_opt_bands, h, w, dtype=torch.float32)
            opt_valid = torch.zeros(h, w, dtype=torch.bool)
            opt_meta: Dict[str, Any] = {"modality": "optical", "id": sample_id}
            optical_available = False
        else:
            opt_img, _opt_mask, opt_meta = self.optical_dataset[opt_idx]
            opt_valid = opt_meta.get(
                "valid_mask", torch.ones(opt_img.shape[-2:], dtype=torch.bool)
            )
            optical_available = True

            # ── spatial shape check ──────────────────────────────────────────
            if self.require_spatial_match:
                sar_hw = (sar_img.shape[-2], sar_img.shape[-1])
                opt_hw = (opt_img.shape[-2], opt_img.shape[-1])
                if sar_hw != opt_hw:
                    raise ValueError(
                        f"SAR and optical spatial dims differ for id='{sample_id}': "
                        f"SAR {sar_hw} vs optical {opt_hw}. "
                        "Either pre-register tiles or set require_spatial_match=False."
                    )

        # ── fuse channels ────────────────────────────────────────────────────
        fused_img = torch.cat([sar_img, opt_img], dim=0)  # [C_sar + C_s2, H, W]

        # ── fuse validity masks ──────────────────────────────────────────────
        fused_valid = sar_valid & opt_valid  # [H, W], bool

        # ── build combined metadata ──────────────────────────────────────────
        metadata: Dict[str, Any] = {
            **sar_meta,
            "optical_img_path": opt_meta.get("img_path"),
            "optical_modality": "optical",
            "fused_optical_available": optical_available,
            "n_sar_bands": sar_img.shape[0],
            "n_optical_bands": opt_img.shape[0],
            # Overwrite valid_mask with the fused one
            "valid_mask": fused_valid,
        }

        return fused_img, mask_tensor, metadata

    def _infer_optical_bands(self) -> int:
        """Return the number of optical channels from the first available sample."""
        if len(self.optical_dataset) == 0:  # type: ignore[arg-type]
            return 1  # safe fallback; will emit zeros anyway
        try:
            opt_img, _, _ = self.optical_dataset[0]
            return opt_img.shape[0]
        except Exception:
            return 1

    def __repr__(self) -> str:
        n_sar = len(self.sar_dataset)  # type: ignore[arg-type]
        n_opt = len(self.optical_dataset)  # type: ignore[arg-type]
        n_paired = sum(
            1 for s in getattr(self.sar_dataset, "samples", [])
            if s["id"] in self._optical_id_to_idx
        )
        return (
            f"FusedDataset("
            f"n_sar={n_sar}, "
            f"n_optical={n_opt}, "
            f"n_paired={n_paired}, "
            f"policy={self.optical_missing_policy!r})"
        )
