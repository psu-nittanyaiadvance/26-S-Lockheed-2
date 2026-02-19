from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def default_collate(
    batch: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Dict[str, Any]]]:
    """
    Stack image tensors and (optionally) mask tensors, keep metadata as a list.
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

    metas = [b[2] for b in batch]
    return images, mask_batch, metas


def semi_supervised_collate(
    batch: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]],
    missing_mask_value: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Collate labeled and unlabeled samples by filling missing masks with a sentinel value.

    Returns mask_batch as uint8 [B,1,H,W] and metas as a list.
    """
    if not batch:
        raise ValueError("Empty batch")
    if not (0 <= missing_mask_value <= 255):
        raise ValueError("missing_mask_value must be in [0, 255]")

    images = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[2] for b in batch]

    masks: List[torch.Tensor] = []
    for img, mask, _ in batch:
        if img.ndim != 3:
            raise ValueError(f"Expected image [C,H,W], got {tuple(img.shape)}")
        _, h, w = img.shape
        if mask is None:
            filled = torch.full(
                (1, h, w), missing_mask_value, dtype=torch.uint8, device=img.device
            )
            masks.append(filled)
        else:
            if mask.ndim != 3 or mask.shape[0] != 1:
                raise ValueError(f"Expected mask [1,H,W], got {tuple(mask.shape)}")
            masks.append(mask.to(dtype=torch.uint8, device=img.device))

    mask_batch = torch.stack(masks, dim=0)
    return images, mask_batch, metas
