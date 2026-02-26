import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval import evaluate
from UNet.UNetModel import UNet as UNetModel
from data_loader import PatchDataset, SARDataset, list_ids_from_dir, validate_sample_shapes

def tversky_loss(inputs, targets, alpha, beta, epsilon=1e-6):
    # 1. Apply sigmoid (binary) or softmax (multiclass) to get probabilities
    inputs = torch.sigmoid(inputs) if inputs.shape[1] == 1 else F.softmax(inputs, dim=1)
    
    # 2. Flatten ONLY the spatial dimensions (Height x Width)
    # This preserves the Batch (dim 0) and Class (dim 1) boundaries
    # Shape changes from [Batch, Classes, Height, Width] -> [Batch, Classes, Pixels]
    inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
    targets = targets.view(targets.shape[0], targets.shape[1], -1)
    
    # 3. Calculate True Positives, False Positives, and False Negatives
    # We sum across dim=2 (the flattened pixels) to get totals PER CLASS
    TP = (inputs * targets).sum(dim=2)    
    FP = ((1 - targets) * inputs).sum(dim=2)
    FN = (targets * (1 - inputs)).sum(dim=2)
    
    # 4. Calculate the Tversky index (Yields a score for each class, per image)
    tversky_index = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
    
    # 5. Average the scores across all classes and batches, then subtract from 1
    return 1 - tversky_index.mean()

def _extract_ignore_mask(metas, device, masks):
    if masks is None or metas is None:
        return None
    if len(metas) == 0:
        return None
    h, w = masks.shape[-2:]
    ignore_masks = []
    found_any = False
    for meta in metas:
        ignore = meta.get("ignore_mask", None)
        if ignore is None:
            ignore_masks.append(torch.zeros((1, h, w), dtype=torch.bool, device=device))
            continue
        found_any = True
        if not isinstance(ignore, torch.Tensor):
            ignore = torch.as_tensor(ignore)
        if ignore.ndim == 2:
            ignore = ignore.unsqueeze(0)
        elif ignore.ndim != 3:
            raise ValueError("ignore_mask must have shape [H,W] or [1,H,W]")
        if ignore.shape[0] != 1:
            raise ValueError("ignore_mask must have shape [1,H,W]")
        if tuple(ignore.shape[-2:]) != (h, w):
            raise ValueError(
                f"ignore_mask spatial shape mismatch: got {tuple(ignore.shape[-2:])}, "
                f"expected {(h, w)}"
            )
        ignore_masks.append(ignore.to(device=device, dtype=torch.bool))
    if not found_any:
        return None
    return torch.stack(ignore_masks, dim=0)

def _apply_ignore_index(targets, ignore_mask, ignore_index):
    if ignore_mask is None:
        return targets
    if not isinstance(ignore_mask, torch.Tensor):
        ignore_mask = torch.as_tensor(ignore_mask)
    if ignore_mask.ndim == targets.ndim + 1:
        ignore_mask = ignore_mask.squeeze(1)
    if ignore_mask.shape != targets.shape:
        raise ValueError(
            f"ignore_mask shape mismatch: got {tuple(ignore_mask.shape)}, "
            f"expected {tuple(targets.shape)}"
        )
    masked = targets.clone()
    masked[ignore_mask.bool()] = ignore_index
    return masked

def init_metric_state(n_classes):
    if n_classes == 1:
        return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "n_classes": 1}
    return {
        "tp": [0.0] * n_classes,
        "fp": [0.0] * n_classes,
        "fn": [0.0] * n_classes,
        "tn": [0.0] * n_classes,
        "n_classes": n_classes,
    }

def update_metric_state(state, logits, masks, n_classes, ignore_mask=None):
    if n_classes == 1:
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks_bin = masks.squeeze(1)
        else:
            masks_bin = masks
        probs = torch.sigmoid(logits)
        if probs.ndim == 4 and probs.shape[1] == 1:
            probs = probs.squeeze(1)
        preds = (probs > 0.5)
        valid = torch.ones_like(masks_bin, dtype=torch.bool)
        if ignore_mask is not None:
            if ignore_mask.ndim == 4 and ignore_mask.shape[1] == 1:
                ignore_mask = ignore_mask.squeeze(1)
            valid = ~ignore_mask.bool()
        true = masks_bin.bool()
        pred = preds.bool()
        tp = (pred & true & valid).sum().item()
        fp = (pred & ~true & valid).sum().item()
        fn = (~pred & true & valid).sum().item()
        tn = (~pred & ~true & valid).sum().item()
        state["tp"] += tp
        state["fp"] += fp
        state["fn"] += fn
        state["tn"] += tn
        return

    if masks.ndim == 4 and masks.shape[1] == 1:
        true = masks.squeeze(1).long()
    else:
        true = masks.long()
    pred = logits.argmax(dim=1)
    valid = torch.ones_like(true, dtype=torch.bool)
    if ignore_mask is not None:
        if ignore_mask.ndim == 4 and ignore_mask.shape[1] == 1:
            ignore_mask = ignore_mask.squeeze(1)
        valid = ~ignore_mask.bool()
    for cl in range(n_classes):
        pred_cl = (pred == cl)
        true_cl = (true == cl)
        tp = (pred_cl & true_cl & valid).sum().item()
        fp = (pred_cl & ~true_cl & valid).sum().item()
        fn = (~pred_cl & true_cl & valid).sum().item()
        tn = (~pred_cl & ~true_cl & valid).sum().item()
        state["tp"][cl] += tp
        state["fp"][cl] += fp
        state["fn"][cl] += fn
        state["tn"][cl] += tn

def finalize_metrics(state, n_classes):
    if n_classes == 1:
        tp = state["tp"]
        fp = state["fp"]
        fn = state["fn"]
        union = tp + fp + fn
        if union == 0:
            iou = float("nan")
            dice = float("nan")
        else:
            iou = tp / union
            denom = (2 * tp + fp + fn)
            dice = float("nan") if denom == 0 else (2 * tp / denom)
        return {"iou": iou, "dice": dice}

    ious = []
    dices = []
    for cl in range(n_classes):
        tp = state["tp"][cl]
        fp = state["fp"][cl]
        fn = state["fn"][cl]
        union = tp + fp + fn
        if union == 0:
            iou = float("nan")
            dice = float("nan")
        else:
            iou = tp / union
            denom = (2 * tp + fp + fn)
            dice = float("nan") if denom == 0 else (2 * tp / denom)
        ious.append(iou)
        dices.append(dice)
    valid_ious = [v for v in ious if v == v]
    valid_dices = [v for v in dices if v == v]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float("nan")
    mean_dice = sum(valid_dices) / len(valid_dices) if valid_dices else float("nan")
    return {"iou": mean_iou, "dice": mean_dice, "iou_per_class": ious, "dice_per_class": dices}

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        tv_alpha = 0.7, #confirm default for Tversky alpha
        tv_beta = 0.3, #confirm default for Tversky beta
        adam_betas = (0.9, 0.999),
        n_classes = 1
        ):

    #optimizer setup
    optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    betas=adam_betas, 
    weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    writer = SummaryWriter(comment=f'LR_{learning_rate}_BS_{batch_size}')
    logging.info(f"Hyperparameters: {locals()}")

    #define loss function with Tversky
    criterion = lambda inputs, targets: tversky_loss(inputs, targets, alpha=tv_alpha, beta=tv_beta)

    grad_scaler = torch.amp.GradScaler(device=device.type, enabled=amp)
    global_step = 0

    for epoch in range(1, epochs + 1):
        #model into training mode
        model.train()
        epoch_loss = 0

        #create progress bar as a 
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as progress_bar:
            for batch in train_loader:
                images, true_masks, valid_mask = batch['image'], batch['mask'], batch['valid_mask']

                #check for correct dimensionality
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                #format and load to device
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                valid_mask = valid_mask.to(device=device)

                #forward pass
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    if model.n_classes == 1:
                        # Add a channel dimension to true_masks so it matches masks_pred shape [Batch, 1, Height, Width]
                        target_masks = true_masks.unsqueeze(1).float()
                        loss_px = criterion(masks_pred, target_masks)
                    else:
                        # One-hot encode the target masks and rearrange dimensions to match masks_pred
                        target_masks = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                        loss_px = criterion(masks_pred, target_masks)
                    vm = valid_mask
                    if loss_px.ndim == 4 and vm.ndim == 3:
                        vm = vm.unsqueeze(1)
                    loss = (loss_px * vm).sum() / vm.sum().clamp(1)
                        
                #clear previous gradients
                optimizer.zero_grad(set_to_none=True)
                #scale gradient to avoid 0 rounded grad values
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                #clip gradient values to maximum
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                progress_bar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Epoch', epoch, global_step)
                progress_bar.set_postfix(**{'loss (batch)': loss.item()})

                # --- Evaluation round (Local Logging Version) ---
                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and val_loader is not None:
                    if global_step % division_step == 0:
                        # 1. Log Weights and Gradients Histograms
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                writer.add_histogram(f'Weights/{tag}', value.data.cpu(), global_step)
                            if value.grad is not None:
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    writer.add_histogram(f'Gradients/{tag}', value.grad.data.cpu(), global_step)

                        # 2. Run Evaluation
                        val_score = evaluate(model, val_loader, device, amp, criterion, n_classes=args.classes) 

                        # val_score is a dict from evaluate()
                        for k, v in val_score.items():
                            # TensorBoard expects numeric scalars; skip non-scalars defensively
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f'Validation/{k}', v, global_step)
                                
                        scheduler.step(val_score["val_mIoU"])

                        # 3. Log Scalars and Images to TensorBoard
                        try:
                            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                            # This groups them together in the UI
                            writer.add_scalars('Validation/Flood_Metrics', {
                                'Precision': val_score['val_flood_precision'],
                                'Recall': val_score['val_flood_recall'],
                                'F1': val_score['val_flood_f1']
                                }, global_step)
                            
                            # Log the first image in the batch
                            # Note: TensorBoard expects (C, H, W)
                            writer.add_image('Visuals/Image', images[0].cpu(), global_step)
                            
                            # Ground Truth Mask (adding channel dim)
                            writer.add_image('Visuals/Mask_True', true_masks[0].float().cpu().unsqueeze(0), global_step)
                            
                            # Predicted Mask (taking argmax and adding channel dim)
                            pred_mask = masks_pred.argmax(dim=1)[0].float().cpu().unsqueeze(0)
                            writer.add_image('Visuals/Mask_Pred', pred_mask, global_step)
                        except Exception as e:
                            logging.warning(f"Could not log to TensorBoard: {e}")
        print(
            f"Validation Results:\n"
            f"  Loss:      {val_score['val_loss']:.4f}\n"
            f"  Accuracy:  {val_score['val_accuracy']:.4f}\n"
            f"  mIoU:      {val_score['val_mIoU']:.4f}\n"
            f"  Flood Metrics -> "
            f"IoU: {val_score['val_flood_iou']:.4f} | "
            f"Prec: {val_score['val_flood_precision']:.4f} | "
            f"Recall: {val_score['val_flood_recall']:.4f} | "
            f"F1: {val_score['val_flood_f1']:.4f}"
        )
        
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


# This argparse block defines command-line options so you can run training with different
# settings (epochs, batch size, LR, scaling, AMP, etc.) without editing the source code.
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--img-dir', type=str, required=True, help='Path to SAR image .tif/.tiff files')
    parser.add_argument('--mask-dir', type=str, required=True, help='Path to mask .tif/.tiff files')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker count')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    return parser.parse_args()



#argparse usage in main
if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetModel(n_channels=2, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    seed = getattr(args, 'seed', 0)
    torch.manual_seed(seed)
    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    mask_id_suffix_map = {'S1Hand': 'S1OtsuLabelHand'}
    ids = list_ids_from_dir(args.img_dir, recursive=True)
    paired_ids_list = []
    missing_in_masks = 0
    for image_id in ids:
        mask_id = image_id
        if image_id.endswith("S1Hand"):
            mask_id = f"{image_id[:-len('S1Hand')]}S1OtsuLabelHand"
        mask_tif = mask_dir / f"{mask_id}.tif"
        mask_tiff = mask_dir / f"{mask_id}.tiff"
        if mask_tif.exists() or mask_tiff.exists():
            paired_ids_list.append(image_id)
        else:
            missing_in_masks += 1
    logging.info(
        "Paired ID report: images_total=%s paired_total=%s missing_in_masks=%s",
        len(ids),
        len(paired_ids_list),
        missing_in_masks,
    )
    if len(paired_ids_list) == 0:
        raise ValueError("No paired image/mask IDs found; aborting training.")
    base_dataset = SARDataset(
        img_root=img_dir,
        mask_root=mask_dir,
        ids_or_paths=paired_ids_list,
        mode='strong',
        mask_id_suffix_map=mask_id_suffix_map,
        expected_img_bands=2,
    )
    base_dataset.mask_values = [0, 1]
    validate_sample_shapes(base_dataset, n=64)
    val_percent = args.val / 100
    n_val = int(len(base_dataset) * val_percent)
    n_train = len(base_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_base, val_base = random_split(base_dataset, [n_train, n_val], generator=generator)
    train_set = PatchDataset(train_base, patch_size=256, overlap=0.2)
    val_set = PatchDataset(val_base, patch_size=256, overlap=0.2) if n_val > 0 else None
    dataset = base_dataset
    n_train = len(train_set)

    def _dict_collate(batch):
        images = torch.stack([b[0] for b in batch], dim=0)
        masks = torch.stack([b[1] for b in batch], dim=0)
        valid_masks = []
        for b in batch:
            meta = b[2]
            vm = meta.get("valid_mask", None)
            if vm is None:
                raise KeyError("Sample metadata is missing 'valid_mask'.")
            if not isinstance(vm, torch.Tensor):
                vm = torch.as_tensor(vm)
            if vm.shape[-2:] != b[0].shape[-2:]:
                y0 = meta.get("patch_y0", None)
                x0 = meta.get("patch_x0", None)
                ps = meta.get("patch_size", None)
                if y0 is not None and x0 is not None and ps is not None:
                    if vm.ndim == 2:
                        vm = vm[y0 : y0 + ps, x0 : x0 + ps]
                    elif vm.ndim == 3:
                        vm = vm[:, y0 : y0 + ps, x0 : x0 + ps]
            valid_masks.append(vm)
        valid_masks = torch.stack(valid_masks, dim=0)
        return {'image': images, 'mask': masks, 'valid_mask': valid_masks}

    num_workers = getattr(args, 'num_workers', 0)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_dict_collate
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_dict_collate
        )
    dir_checkpoint = Path('checkpoints')
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            n_classes=args.classes
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            n_classes=args.classes
        )
