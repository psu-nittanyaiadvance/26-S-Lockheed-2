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

    #stack images and masks, keep metadata as a list
    images = torch.stack([b[0] for b in batch], dim=0)
    masks = [b[1] for b in batch]

    if all(m is None for m in masks):
        mask_batch = None
    elif any(m is None for m in masks): #checking for mixed labeled and unlabeled samples in the same batch
        raise ValueError("Mixed labeled and unlabeled samples in the same batch")
    else:
        mask_batch = torch.stack([m for m in masks], dim=0)

    metas = [b[2] for b in batch]
    return images, mask_batch, metas
