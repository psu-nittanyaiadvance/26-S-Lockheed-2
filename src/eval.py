"""
@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion, n_classes):
    
    #Active supported mode: Phase 1 SAR-only binary flood segmentation
    #(1 = flood, 0 = background, valid_mask is the only ignore mechanism).
    
    require_active_binary_mode(n_classes, "evaluate()")

    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0.0
    tp = torch.tensor(0.0, device=device)
    fp = torch.tensor(0.0, device=device)
    fn = torch.tensor(0.0, device=device)
    tn = torch.tensor(0.0, device=device)
    eps = 1e-6

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation round", unit="batch", leave=False):
            image = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = batch["mask"].to(device=device)

            mask_pred = net(image)
            target_masks = prepare_binary_target(mask_true).to(device=device)
            valid_mask = prepare_binary_valid_mask(batch.get("valid_mask", None), mask_pred)

            total_loss += criterion(mask_pred, target_masks, valid_mask).item()

            confusion = compute_binary_confusion(
                mask_pred,
                target_masks,
                valid_mask=valid_mask,
                threshold=ACTIVE_BINARY_METRIC_THRESHOLD,
            )
            tp += confusion["tp"]
            fp += confusion["fp"]
            fn += confusion["fn"]
            tn += confusion["tn"]

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    flood_iou = tp / (tp + fp + fn + eps)
    land_iou = tn / (tn + fp + fn + eps)
    miou = 0.5 * (flood_iou + land_iou)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    net.train()
    return {
        "val_loss": total_loss / max(num_val_batches, 1),
        "val_accuracy": accuracy.item(),
        "val_mIoU": miou.item(),
        "val_flood_iou": flood_iou.item(),
        "val_flood_precision": precision.item(),
        "val_flood_recall": recall.item(),
        "val_flood_f1": f1.item(),
    }
"""

import torch
from tqdm import tqdm

from binary_mode import (
    ACTIVE_BINARY_METRIC_THRESHOLD,
    compute_binary_confusion,
    prepare_binary_target,
    prepare_binary_valid_mask,
    require_active_binary_mode,
)


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion, n_classes):
    """
    Active supported mode: Phase 1 SAR-only binary flood segmentation
    (1 = flood, 0 = background, valid_mask is the only ignore mechanism).
    """
    require_active_binary_mode(n_classes, "evaluate()")

    was_training = net.training
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0.0
    tp = torch.tensor(0.0, device=device)
    fp = torch.tensor(0.0, device=device)
    fn = torch.tensor(0.0, device=device)
    tn = torch.tensor(0.0, device=device)
    eps = 1e-6

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation round", unit="batch", leave=False):
            image = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = batch["mask"].to(device=device)

            # Raw Logits - Shape remains [B, 1, H, W]
            mask_pred = net(image) 
            target_masks = prepare_binary_target(mask_true).to(device=device)
            valid_mask = prepare_binary_valid_mask(batch.get("valid_mask", None), mask_pred)

            # Calculate Loss
            total_loss += criterion(mask_pred, target_masks, valid_mask).item()

            # Pass the raw un-squeezed logits directly. 
            # binary_mode.py expects [B, 1, H, W] and handles the sigmoid internally.
            confusion = compute_binary_confusion(
                mask_pred,
                target_masks,
                valid_mask=valid_mask,
                threshold=ACTIVE_BINARY_METRIC_THRESHOLD,
            )
            
            tp += confusion["tp"]
            fp += confusion["fp"]
            fn += confusion["fn"]
            tn += confusion["tn"]

    # Global Metric Calculation
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    flood_iou = tp / (tp + fp + fn + eps)
    land_iou = tn / (tn + fp + fn + eps)
    miou = 0.5 * (flood_iou + land_iou)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    if was_training:
        net.train()
    return {
        "val_loss": total_loss / max(num_val_batches, 1),
        "val_accuracy": accuracy.item(),
        "val_mIoU": miou.item(),
        "val_flood_iou": flood_iou.item(),
        "val_flood_precision": precision.item(),
        "val_flood_recall": recall.item(),
        "val_flood_f1": f1.item(),
    }
