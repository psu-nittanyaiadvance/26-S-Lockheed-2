import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion, n_classes):
    net.eval()
    num_val_batches = len(dataloader)
    
    # Initialize accumulators for metrics
    total_loss = 0
    total_tp = torch.zeros(n_classes).to(device)
    total_fp = torch.zeros(n_classes).to(device)
    total_fn = torch.zeros(n_classes).to(device)
    total_tn = torch.zeros(n_classes).to(device)
    eps = 1e-6

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move to device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 1. Forward Pass & Loss (Matching your training logic)
            mask_pred = net(image)
            
            if n_classes == 1:
                target_masks = mask_true.unsqueeze(1).float()
            else:
                target_masks = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            
            total_loss += criterion(mask_pred, target_masks).item()

            # 2. Get Hard Predictions for Metrics
            if n_classes == 1:
                pred_labels = (torch.sigmoid(mask_pred) > 0.5).long()
            else:
                pred_labels = mask_pred.argmax(dim=1)

            # 3. Calculate TP, FP, FN, TN per class
            for cl in range(n_classes):
                true_class = (mask_true == cl)
                pred_class = (pred_labels == cl) if n_classes > 1 else (pred_labels.squeeze(1) == cl)
                
                total_tp[cl] += (pred_class & true_class).sum()
                total_fp[cl] += (pred_class & ~true_class).sum()
                total_fn[cl] += (~pred_class & true_class).sum()
                total_tn[cl] += (~pred_class & ~true_class).sum()

    # --- Calculate Final Metrics ---
    # Per-class metrics
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    
    # Accuracy is global
    total_pixels = (total_tp + total_fp + total_fn + total_tn).sum() / n_classes
    accuracy = (total_tp.sum() + total_tn.sum()) / (total_pixels * n_classes + eps)
    miou = iou.mean().item()

    net.train() # Reset to training mode

    # Return dictionary for easy logging
    # For Flood (Class 1), we focus specifically on those values
    return {
        'val_loss': total_loss / max(num_val_batches, 1),
        'val_accuracy': accuracy.item(),
        'val_mIoU': miou,
        'val_flood_iou': iou[1].item() if n_classes > 1 else iou[0].item(),
        'val_flood_precision': precision[1].item() if n_classes > 1 else precision[0].item(),
        'val_flood_recall': recall[1].item() if n_classes > 1 else recall[0].item(),
        'val_flood_f1': f1[1].item() if n_classes > 1 else f1[0].item()
    }