import torch
import torch.nn.functional as F
from tqdm import tqdm
'''
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
            valid_mask = batch.get('valid_mask', None)


            # Move to device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            if valid_mask is not None:
                valid_mask = valid_mask.to(device=device).bool()

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
    land_iou = total_tn / (total_tn + total_fp + total_fn + eps)

    miou = (iou[0].item() + land_iou[0].item()) / 2
    
    # Accuracy is global
    total_pixels = (total_tp + total_fp + total_fn + total_tn).sum() / n_classes
    accuracy = (total_tp.sum() + total_tn.sum()) / (total_pixels * n_classes + eps)
    

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
    '''


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion, n_classes):
    net.eval()
    num_val_batches = len(dataloader)

    total_loss = 0.0
    eps = 1e-6

    if n_classes == 1:
        tp = torch.tensor(0.0, device=device)
        fp = torch.tensor(0.0, device=device)
        fn = torch.tensor(0.0, device=device)
        tn = torch.tensor(0.0, device=device)
    else:
        total_tp = torch.zeros(n_classes, device=device)
        total_fp = torch.zeros(n_classes, device=device)
        total_fn = torch.zeros(n_classes, device=device)
        total_tn = torch.zeros(n_classes, device=device)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            valid_mask = batch.get('valid_mask', None)

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            if valid_mask is not None:
                valid_mask = valid_mask.to(device=device).bool()

            mask_pred = net(image)

            # Loss (same as train)
            if n_classes == 1:
                target_masks = mask_true.unsqueeze(1).float()
            else:
                target_masks = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()

            total_loss += criterion(mask_pred, target_masks, valid_mask).item()

            # Hard preds
            if n_classes == 1:
                pred_pos = (torch.sigmoid(mask_pred).squeeze(1) > 0.35)   # [B,H,W] bool
                true_pos = (mask_true == 1)                              # flood is 1

                if valid_mask is not None:
                    pred_pos = pred_pos & valid_mask
                    true_pos = true_pos & valid_mask
                    vm = valid_mask
                else:
                    vm = torch.ones_like(true_pos, dtype=torch.bool)

                # Compute confusion counts over valid pixels
                tp += (pred_pos & true_pos).sum()
                fp += (pred_pos & ~true_pos & vm).sum()
                fn += (~pred_pos & true_pos).sum()
                tn += (~pred_pos & ~true_pos & vm).sum()

            else:
                pred_labels = mask_pred.argmax(dim=1)  # [B,H,W]
                for cl in range(n_classes):
                    true_class = (mask_true == cl)
                    pred_class = (pred_labels == cl)

                    if valid_mask is not None:
                        true_class = true_class & valid_mask
                        pred_class = pred_class & valid_mask
                        vm = valid_mask
                    else:
                        vm = torch.ones_like(true_class, dtype=torch.bool)

                    total_tp[cl] += (pred_class & true_class).sum()
                    total_fp[cl] += (pred_class & ~true_class & vm).sum()
                    total_fn[cl] += (~pred_class & true_class).sum()
                    total_tn[cl] += (~pred_class & ~true_class & vm).sum()

    # Final metrics
    if n_classes == 1:
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * (precision * recall) / (precision + recall + eps)

        flood_iou = tp / (tp + fp + fn + eps)
        land_iou  = tn / (tn + fp + fn + eps)          # IoU of background
        miou      = 0.5 * (flood_iou + land_iou)

        accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)

        net.train()
        return {
            'val_loss': total_loss / max(num_val_batches, 1),
            'val_accuracy': accuracy.item(),
            'val_mIoU': miou.item(),
            'val_flood_iou': flood_iou.item(),
            'val_flood_precision': precision.item(),
            'val_flood_recall': recall.item(),
            'val_flood_f1': f1.item(),
        }

    else:
        precision = total_tp / (total_tp + total_fp + eps)
        recall    = total_tp / (total_tp + total_fn + eps)
        f1        = 2 * (precision * recall) / (precision + recall + eps)
        iou       = total_tp / (total_tp + total_fp + total_fn + eps)
        miou      = iou.mean().item()

        total_pixels = (total_tp + total_fp + total_fn + total_tn).sum()
        accuracy = (total_tp.sum() + total_tn.sum()) / (total_pixels + eps)

        net.train()
        return {
            'val_loss': total_loss / max(num_val_batches, 1),
            'val_accuracy': accuracy.item(),
            'val_mIoU': miou,
            'val_flood_iou': iou[1].item(),
            'val_flood_precision': precision[1].item(),
            'val_flood_recall': recall[1].item(),
            'val_flood_f1': f1[1].item(),
        }