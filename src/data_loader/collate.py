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
