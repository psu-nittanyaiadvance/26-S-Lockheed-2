from __future__ import annotations

from typing import Any, Dict, Optional

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


def _as_hw_mask(valid_mask: Optional[torch.Tensor], hw: torch.Size) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(hw, dtype=torch.bool)
    if not isinstance(valid_mask, torch.Tensor):
        valid_mask = torch.as_tensor(valid_mask)
    if valid_mask.ndim == 3:
        if valid_mask.shape[0] != 1:
            raise ValueError(f"valid_mask must have shape [H,W] or [1,H,W], got {tuple(valid_mask.shape)}")
        valid_mask = valid_mask.squeeze(0)
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape [H,W], got {tuple(valid_mask.shape)}")
    if tuple(valid_mask.shape) != tuple(hw):
        raise ValueError(f"valid_mask shape mismatch: got {tuple(valid_mask.shape)}, expected {tuple(hw)}")
    return valid_mask.to(dtype=torch.bool)


def _mask_invalid_pixels(image: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return image * valid_mask.unsqueeze(0).to(dtype=image.dtype)


def build_decur_batch_views(
    batch: Dict[str, Any],
    *,
    sar_transform,
    opt_transform,
) -> Dict[str, torch.Tensor]:
    required = {"sar", "optical", "valid_mask", "meta"}
    missing = sorted(name for name in required if name not in batch)
    if missing:
        raise KeyError(f"Paired multimodal batch missing required keys: {missing}")

    sar_batch = batch["sar"]
    optical_batch = batch["optical"]
    valid_mask_batch = batch["valid_mask"]
    metas = batch["meta"]

    if not isinstance(sar_batch, torch.Tensor) or sar_batch.ndim != 4:
        raise ValueError(f"batch['sar'] must have shape [B,C,H,W], got {type(sar_batch)!r} {tuple(getattr(sar_batch, 'shape', ())) }")
    if not isinstance(optical_batch, torch.Tensor) or optical_batch.ndim != 4:
        raise ValueError(f"batch['optical'] must have shape [B,C,H,W], got {type(optical_batch)!r} {tuple(getattr(optical_batch, 'shape', ())) }")
    if not isinstance(valid_mask_batch, torch.Tensor) or valid_mask_batch.ndim != 4:
        raise ValueError(
            f"batch['valid_mask'] must have shape [B,1,H,W], got {type(valid_mask_batch)!r} {tuple(getattr(valid_mask_batch, 'shape', ())) }"
        )
    if len(metas) != sar_batch.shape[0]:
        raise ValueError(f"batch['meta'] length mismatch: got {len(metas)}, expected {sar_batch.shape[0]}")
    if sar_batch.shape[0] != optical_batch.shape[0] or sar_batch.shape[0] != valid_mask_batch.shape[0]:
        raise ValueError("Batch size mismatch among 'sar', 'optical', and 'valid_mask'")

    views = {
        "sar_view_1": [],
        "sar_view_2": [],
        "opt_view_1": [],
        "opt_view_2": [],
        "sar_valid_1": [],
        "sar_valid_2": [],
        "opt_valid_1": [],
        "opt_valid_2": [],
    }

    for sar_image, optical_image, valid_mask, meta in zip(sar_batch, optical_batch, valid_mask_batch, metas):
        if tuple(sar_image.shape[-2:]) != tuple(optical_image.shape[-2:]):
            raise ValueError(
                "SAR and optical spatial dims must match within each sample; "
                f"got SAR {tuple(sar_image.shape)} and optical {tuple(optical_image.shape)}"
            )

        base_valid_mask = _as_hw_mask(valid_mask, sar_image.shape[-2:])
        base_meta = dict(meta)

        sar_meta = dict(base_meta)
        sar_meta["valid_mask"] = base_valid_mask.clone()
        sar_meta["modality"] = "sar"

        opt_meta = dict(base_meta)
        opt_meta["valid_mask"] = base_valid_mask.clone()
        opt_meta["modality"] = "optical"

        sar_view_1, _, sar_meta_1 = sar_transform(sar_image.clone(), None, dict(sar_meta))
        sar_view_2, _, sar_meta_2 = sar_transform(sar_image.clone(), None, dict(sar_meta))
        opt_view_1, _, opt_meta_1 = opt_transform(optical_image.clone(), None, dict(opt_meta))
        opt_view_2, _, opt_meta_2 = opt_transform(optical_image.clone(), None, dict(opt_meta))

        sar_valid_1 = _as_hw_mask(sar_meta_1.get("valid_mask"), sar_view_1.shape[-2:])
        sar_valid_2 = _as_hw_mask(sar_meta_2.get("valid_mask"), sar_view_2.shape[-2:])
        opt_valid_1 = _as_hw_mask(opt_meta_1.get("valid_mask"), opt_view_1.shape[-2:])
        opt_valid_2 = _as_hw_mask(opt_meta_2.get("valid_mask"), opt_view_2.shape[-2:])

        views["sar_view_1"].append(_mask_invalid_pixels(sar_view_1, sar_valid_1))
        views["sar_view_2"].append(_mask_invalid_pixels(sar_view_2, sar_valid_2))
        views["opt_view_1"].append(_mask_invalid_pixels(opt_view_1, opt_valid_1))
        views["opt_view_2"].append(_mask_invalid_pixels(opt_view_2, opt_valid_2))
        views["sar_valid_1"].append(sar_valid_1)
        views["sar_valid_2"].append(sar_valid_2)
        views["opt_valid_1"].append(opt_valid_1)
        views["opt_valid_2"].append(opt_valid_2)

    return {key: torch.stack(value, dim=0) for key, value in views.items()}


@torch.inference_mode()
def evaluate_decur(
    model: torch.nn.Module,
    dataloader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    amp: bool,
    common_dim: int,
    sar_transform,
    opt_transform,
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
        views = build_decur_batch_views(
            batch,
            sar_transform=sar_transform,
            opt_transform=opt_transform,
        )
        views = _move_batch_to_device(views, device)

        with torch.autocast(device_type=autocast_device, enabled=amp):
            z_sar_1, z_sar_2, z_opt_1, z_opt_2 = model(
                views["sar_view_1"],
                views["sar_view_2"],
                views["opt_view_1"],
                views["opt_view_2"],
            )
            loss = loss_fn(z_sar_1, z_sar_2, z_opt_1, z_opt_2)

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
