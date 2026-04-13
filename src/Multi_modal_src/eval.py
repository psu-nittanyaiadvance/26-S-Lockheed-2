from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _move_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_batch_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_batch_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(v, device) for v in value)
    return value


@torch.inference_mode()
def evaluate_decur(
    model: torch.nn.Module,
    dataloader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    amp: bool,
    common_dim: int,
) -> Dict[str, float]:
    # Validation uses the same four-view forward pass as training, but only
    # reads the resulting embeddings to compute loss and alignment metrics.
    was_training = model.training
    model.eval()

    totals = {
        "val_loss": 0.0,
        "val_cross_modal_cosine_full": 0.0,
        "val_cross_modal_cosine_common": 0.0,
        "val_sar_view_consistency": 0.0,
        "val_optical_view_consistency": 0.0,
    }
    num_batches = 0
    autocast_device = device.type if device.type != "mps" else "cpu"

    for batch in tqdm(dataloader, desc="Validation", unit="batch", leave=False):
        batch = _move_batch_to_device(batch, device)

        with torch.autocast(device_type=autocast_device, enabled=amp):
            z_sar_1, z_sar_2, z_opt_1, z_opt_2 = model(
                batch["sar_view_1"],
                batch["sar_view_2"],
                batch["opt_view_1"],
                batch["opt_view_2"],
            )

            #indexed to only include the loss value
            loss = loss_fn(z_sar_1, z_sar_2, z_opt_1, z_opt_2)[0]

        # These metrics are simple but readable:
        # full/common cross-modal cosine tells us whether SAR and optical
        # embeddings are meeting in the shared space, while view consistency
        # tells us whether augmentations preserve representation identity.
        dim_common = min(common_dim, z_sar_1.shape[1], z_opt_1.shape[1])
        totals["val_loss"] += float(loss.item())
        totals["val_cross_modal_cosine_full"] += float(
            F.cosine_similarity(z_sar_1, z_opt_1, dim=1).mean().item()
        )
        totals["val_cross_modal_cosine_common"] += float(
            F.cosine_similarity(z_sar_1[:, :dim_common], z_opt_1[:, :dim_common], dim=1).mean().item()
        )
        totals["val_sar_view_consistency"] += float(
            F.cosine_similarity(z_sar_1, z_sar_2, dim=1).mean().item()
        )
        totals["val_optical_view_consistency"] += float(
            F.cosine_similarity(z_opt_1, z_opt_2, dim=1).mean().item()
        )
        num_batches += 1

    if was_training:
        model.train()

    if num_batches == 0:
        return {key: float("nan") for key in totals}
    return {key: value / num_batches for key, value in totals.items()}
