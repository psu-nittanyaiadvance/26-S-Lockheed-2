"""
fused_dataset.py - Late-pixel fusion of SAR (S1) and Optical (S2) datasets.

FusedDataset pairs a SARDataset and an OpticalDataset by sample ID and
concatenates their image tensors along the channel dimension:

    [C_sar + C_s2, H, W]

The fused sample follows the SAR dataset contract from the pipeline's
perspective:
- image: float32 tensor [C, H, W]
- mask: SAR mask tensor [1, H, W] or None
- metadata includes id, img_path, mask_path, and valid_mask

Missing optical handling:
- "zeros": zero-fill optical channels and keep the sample trainable by using
  the SAR valid_mask
- "raise": raise if a paired optical tile is unavailable
- "sar_only": accepted for backwards compatibility and aliased to "zeros"
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class FusedDataset(Dataset):
    """
    Combines a SARDataset and an OpticalDataset, aligning samples by ID.

    The combined image tensor has shape ``[C_sar + C_s2, H, W]`` with SAR
    bands first. The mask is passed through from ``sar_dataset`` so unlabeled
    SAR samples remain unlabeled in the fused view as well.

    Parameters
    ----------
    sar_dataset : SARDataset
        Sentinel-1 dataset instance.
    optical_dataset : OpticalDataset
        Sentinel-2 dataset instance.
    optical_missing_policy : {"zeros", "sar_only", "raise"}
        How to handle SAR IDs that have no paired optical tile. ``sar_only``
        is deprecated and behaves identically to ``zeros``.
    require_spatial_match : bool
        If True (default), raise ValueError if SAR and optical spatial dims
        differ at __getitem__ time.
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

        # "sar_only" is deprecated and intentionally aliases to "zeros" so
        # missing-optical handling follows one code path.
        if optical_missing_policy == "sar_only":
            optical_missing_policy = "zeros"

        self.sar_dataset = sar_dataset
        self.optical_dataset = optical_dataset
        self.optical_missing_policy = optical_missing_policy
        self.require_spatial_match = require_spatial_match

        self._optical_id_to_idx: Dict[str, int] = {}
        for i in range(len(optical_dataset)):  # type: ignore[arg-type]
            samples_attr = getattr(optical_dataset, "samples", None)
            if samples_attr is not None:
                optical_id = samples_attr[i]["id"]
            else:
                _, _, meta = optical_dataset[i]
                optical_id = meta.get("id", str(i))
            self._optical_id_to_idx[optical_id] = i

        self._n_optical_bands = self._infer_optical_bands()

        if self.optical_missing_policy == "raise":
            sar_samples = getattr(sar_dataset, "samples", None)
            if sar_samples is not None:
                missing = [
                    sample["id"]
                    for sample in sar_samples
                    if sample["id"] not in self._optical_id_to_idx
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
        sample_id = str(sar_meta.get("id", idx))

        sar_img = sar_img.to(dtype=torch.float32)
        sar_hw = (sar_img.shape[-2], sar_img.shape[-1])
        sar_valid = self._coerce_valid_mask(
            sar_meta.get("valid_mask"),
            sar_hw,
            sample_id=sample_id,
            source="sar",
        )

        opt_idx = self._optical_id_to_idx.get(sample_id)
        if opt_idx is None:
            if self.optical_missing_policy == "raise":
                raise RuntimeError(f"No paired optical tile for SAR id='{sample_id}'")

            h, w = sar_hw
            opt_img = torch.zeros(self._n_optical_bands, h, w, dtype=torch.float32)
            opt_valid = sar_valid
            opt_meta: Dict[str, Any] = {
                "id": sample_id,
                "img_path": None,
                "mask_path": None,
                "valid_mask": opt_valid,
                "modality": "optical",
            }
            optical_available = False
        else:
            opt_img, _opt_mask, opt_meta = self.optical_dataset[opt_idx]
            opt_img = opt_img.to(dtype=torch.float32)
            opt_hw = (opt_img.shape[-2], opt_img.shape[-1])
            opt_valid = self._coerce_valid_mask(
                opt_meta.get("valid_mask"),
                opt_hw,
                sample_id=sample_id,
                source="optical",
            )
            optical_available = True

            if self.require_spatial_match and sar_hw != opt_hw:
                raise ValueError(
                    f"SAR and optical spatial dims differ for id='{sample_id}': "
                    f"SAR {sar_hw} vs optical {opt_hw}. "
                    "Either pre-register tiles or set require_spatial_match=False."
                )

        fused_img = torch.cat([sar_img, opt_img], dim=0)
        fused_valid = sar_valid & opt_valid if optical_available else sar_valid

        metadata = dict(sar_meta)
        metadata["id"] = sample_id
        metadata["img_path"] = sar_meta.get("img_path")
        metadata["mask_path"] = sar_meta.get("mask_path")
        metadata["valid_mask"] = fused_valid
        metadata["modality"] = "fused"
        metadata["optical_img_path"] = opt_meta.get("img_path")
        metadata["fused_optical_available"] = optical_available
        metadata["n_sar_bands"] = sar_img.shape[0]
        metadata["n_optical_bands"] = opt_img.shape[0]

        return fused_img, mask_tensor, metadata

    def _infer_optical_bands(self) -> int:
        """Infer optical channel count once and keep it consistent."""
        if len(self.optical_dataset) == 0:  # type: ignore[arg-type]
            return 1

        for i in range(len(self.optical_dataset)):  # type: ignore[arg-type]
            try:
                opt_img, _, _ = self.optical_dataset[i]
            except Exception:
                continue
            if opt_img.ndim != 3:
                raise ValueError(
                    f"Expected optical image with shape [C,H,W], got {tuple(opt_img.shape)}"
                )
            return int(opt_img.shape[0])

        return 1

    def _coerce_valid_mask(
        self,
        valid_mask: Any,
        expected_hw: Tuple[int, int],
        *,
        sample_id: str,
        source: str,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(expected_hw, dtype=torch.bool)
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = torch.as_tensor(valid_mask)
        if valid_mask.ndim == 3:
            if valid_mask.shape[0] != 1:
                raise ValueError(
                    f"{source} valid_mask must have shape [H,W] or [1,H,W] for id='{sample_id}'"
                )
            valid_mask = valid_mask.squeeze(0)
        if valid_mask.ndim != 2:
            raise ValueError(
                f"{source} valid_mask must have shape [H,W] for id='{sample_id}', "
                f"got {tuple(valid_mask.shape)}"
            )
        if tuple(valid_mask.shape) != tuple(expected_hw):
            raise ValueError(
                f"{source} valid_mask shape mismatch for id='{sample_id}': "
                f"got {tuple(valid_mask.shape)}, expected {tuple(expected_hw)}"
            )
        return valid_mask.to(dtype=torch.bool)

    def __repr__(self) -> str:
        n_sar = len(self.sar_dataset)  # type: ignore[arg-type]
        n_opt = len(self.optical_dataset)  # type: ignore[arg-type]
        n_paired = sum(
            1
            for sample in getattr(self.sar_dataset, "samples", [])
            if sample["id"] in self._optical_id_to_idx
        )
        return (
            f"FusedDataset("
            f"n_sar={n_sar}, "
            f"n_optical={n_opt}, "
            f"n_paired={n_paired}, "
            f"policy={self.optical_missing_policy!r})"
        )
