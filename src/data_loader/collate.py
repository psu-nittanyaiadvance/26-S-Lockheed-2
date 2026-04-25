from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def default_collate(
    batch: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[Dict[str, Any]]]:
    """
    Stack image tensors, (optionally) mask tensors, and valid_mask tensors.
    Metadata dicts are returned as a list with ``valid_mask`` removed (it lives
    in the dedicated ``valid_masks`` batch tensor instead).

    Returns
    -------
    images : torch.Tensor
        Shape ``[B, C, H, W]``, float32.
    masks : torch.Tensor or None
        Shape ``[B, 1, H, W]``, uint8, or ``None`` for unlabeled batches.
    valid_masks : torch.Tensor
        Shape ``[B, H, W]``, bool.  ``False`` wherever any band was NaN/Inf.
        Pass this to your loss function to zero out invalid pixels:

            loss_px = criterion(pred, target)          # reduction='none'
            loss = (loss_px * valid_masks).sum() / valid_masks.sum().clamp(1)

    metas : list of dict
        Per-sample metadata (``valid_mask`` key removed; use the batch tensor).
    """
    if not batch:
        raise ValueError("Empty batch")

    images = torch.stack([b[0] for b in batch], dim=0)

    masks = [b[1] for b in batch]
    if all(m is None for m in masks):
        mask_batch = None
    elif any(m is None for m in masks):
        raise ValueError("Mixed labeled and unlabeled samples in the same batch")
    else:
        mask_batch = torch.stack([m for m in masks], dim=0)

    # Extract valid_mask from each sample's metadata dict and batch it.
    metas: List[Dict[str, Any]] = []
    valid_mask_list = []
    for b in batch:
        meta = dict(b[2])  # shallow copy so we don't mutate the original
        vm = meta.pop("valid_mask", None)
        if vm is None:
            vm = torch.ones(b[0].shape[-2:], dtype=torch.bool)
        if not isinstance(vm, torch.Tensor):
            vm = torch.as_tensor(vm)
        if vm.ndim == 3 and vm.shape[0] == 1:
            vm = vm.squeeze(0)
        valid_mask_list.append(vm)
        metas.append(meta)

    valid_masks = torch.stack(valid_mask_list, dim=0)  # [B, H, W], bool

    return images, mask_batch, valid_masks, metas


def multimodal_pretrain_collate(
    batch: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Collate strict paired multimodal pretraining samples into a dict batch.

    Expected per-sample keys:
    - ``sar``: float tensor [C_sar, H, W]
    - ``optical``: float tensor [C_opt, H, W]
    - ``valid_mask``: bool-like tensor [H, W] or [1, H, W]
    - ``meta``: metadata dict with no label-bearing fields
    """
    if not batch:
        raise ValueError("Empty batch")

    sar_items: List[torch.Tensor] = []
    optical_items: List[torch.Tensor] = []
    valid_masks: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []

    expected_sar_shape: Optional[Tuple[int, int, int]] = None
    expected_optical_shape: Optional[Tuple[int, int, int]] = None

    for idx, sample in enumerate(batch):
        if not isinstance(sample, dict):
            raise TypeError(
                "multimodal_pretrain_collate expects dict samples; "
                f"got {type(sample)!r} at batch index {idx}"
            )

        required = {"sar", "optical", "valid_mask", "meta"}
        missing = sorted(name for name in required if name not in sample)
        if missing:
            raise KeyError(
                f"Missing required keys {missing} in multimodal sample at batch index {idx}"
            )

        sar = sample["sar"]
        optical = sample["optical"]
        valid_mask = sample["valid_mask"]
        meta = sample["meta"]

        if not isinstance(meta, dict):
            raise TypeError(
                f"Sample metadata must be a dict, got {type(meta)!r} at batch index {idx}"
            )
        forbidden_meta = {"mask_path", "label_path"}
        present_forbidden = sorted(name for name in forbidden_meta if name in meta)
        if present_forbidden:
            raise ValueError(
                "Multimodal pretraining batches must not carry label-bearing metadata; "
                f"found {present_forbidden} at batch index {idx}"
            )

        if not isinstance(sar, torch.Tensor):
            sar = torch.as_tensor(sar)
        if not isinstance(optical, torch.Tensor):
            optical = torch.as_tensor(optical)
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = torch.as_tensor(valid_mask)

        if sar.ndim != 3:
            raise ValueError(
                f"SAR sample must have shape [C,H,W], got {tuple(sar.shape)} at batch index {idx}"
            )
        if optical.ndim != 3:
            raise ValueError(
                "Optical sample must have shape [C,H,W], "
                f"got {tuple(optical.shape)} at batch index {idx}"
            )
        if tuple(sar.shape[-2:]) != tuple(optical.shape[-2:]):
            raise ValueError(
                "SAR and optical spatial dims must match within each sample; "
                f"got SAR {tuple(sar.shape)} and optical {tuple(optical.shape)} "
                f"at batch index {idx}"
            )

        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0)
        elif valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
            pass
        else:
            raise ValueError(
                "valid_mask must have shape [H,W] or [1,H,W], "
                f"got {tuple(valid_mask.shape)} at batch index {idx}"
            )
        if tuple(valid_mask.shape[-2:]) != tuple(sar.shape[-2:]):
            raise ValueError(
                "valid_mask spatial dims must match SAR/optical dims; "
                f"got valid_mask {tuple(valid_mask.shape)} and image {tuple(sar.shape)} "
                f"at batch index {idx}"
            )

        if expected_sar_shape is None:
            expected_sar_shape = tuple(sar.shape)
            expected_optical_shape = tuple(optical.shape)
        else:
            if tuple(sar.shape) != expected_sar_shape:
                raise ValueError(
                    "All SAR tensors in a batch must share the same shape; "
                    f"expected {expected_sar_shape}, got {tuple(sar.shape)} "
                    f"at batch index {idx}"
                )
            if tuple(optical.shape) != expected_optical_shape:
                raise ValueError(
                    "All optical tensors in a batch must share the same shape; "
                    f"expected {expected_optical_shape}, got {tuple(optical.shape)} "
                    f"at batch index {idx}"
                )

        sar_items.append(sar.to(dtype=torch.float32))
        optical_items.append(optical.to(dtype=torch.float32))
        valid_masks.append(valid_mask.to(dtype=torch.bool))
        metas.append(dict(meta))

    return {
        "sar": torch.stack(sar_items, dim=0),
        "optical": torch.stack(optical_items, dim=0),
        "valid_mask": torch.stack(valid_masks, dim=0),
        "meta": metas,
    }
