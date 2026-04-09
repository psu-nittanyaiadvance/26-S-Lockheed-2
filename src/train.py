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

from binary_mode import (
    ACTIVE_BINARY_METRIC_THRESHOLD,
    compute_binary_confusion,
    prepare_binary_target,
    prepare_binary_valid_mask,
    require_active_binary_mode,
)
from eval import evaluate
from UNet.UNetModel import UNet as UNetModel
from data_loader import (
    FusedDataset,
    PatchDataset,
    SARDataset,
    list_ids_from_dir,
    multimodal_pretrain_collate,
    validate_sample_shapes,
)


DEFAULT_MASK_ID_SUFFIX_MAP = {
    "S1Hand": "S1OtsuLabelHand",
    "S1Weak": "S1OtsuLabelWeak",
}


def resolve_mask_id(image_id, mask_dir, mask_id_suffix_map):
    direct_tif = mask_dir / f"{image_id}.tif"
    direct_tiff = mask_dir / f"{image_id}.tiff"
    if direct_tif.exists() or direct_tiff.exists():
        return image_id

    for img_suffix, mask_suffix in mask_id_suffix_map.items():
        if image_id.endswith(img_suffix):
            mask_id = f"{image_id[:-len(img_suffix)]}{mask_suffix}"
            mask_tif = mask_dir / f"{mask_id}.tif"
            mask_tiff = mask_dir / f"{mask_id}.tiff"
            if mask_tif.exists() or mask_tiff.exists():
                return mask_id

    return None


# Standard focal-Tversky uses (1 - TI)^gamma. Keeping gamma at 4/3 preserves
# the historical active-path loss curve after correcting gamma semantics.
def focal_tversky_loss(inputs, targets, alpha, beta, gamma=4.0 / 3.0, valid_mask=None, epsilon=1e-6):
    # 1. Apply sigmoid (binary) or softmax (multiclass) to get probabilities
    inputs = torch.sigmoid(inputs) if inputs.shape[1] == 1 else F.softmax(inputs, dim=1)

    if inputs.shape[1] == 1:
        targets = prepare_binary_target(targets).to(device=inputs.device, dtype=inputs.dtype)
        valid_mask = prepare_binary_valid_mask(valid_mask, inputs)
    
    # 2. Flatten ONLY the spatial dimensions (Height x Width)
    # This preserves the Batch (dim 0) and Class (dim 1) boundaries
    # Shape changes from [Batch, Classes, Height, Width] -> [Batch, Classes, Pixels]
    inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
    targets = targets.view(targets.shape[0], targets.shape[1], -1)

    if valid_mask is not None:
        # Force valid_mask to align with [B, C, Pixels]
        # Accepts [B,H,W] or [B,1,H,W] and converts to [B,1,Pixels]
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)  # [B,1,H,W]
        # Now flatten spatial dims to Pixels using the SAME Pixels length as inputs
        valid_mask = valid_mask.reshape(inputs.shape[0], 1, inputs.shape[2]).to(inputs.dtype)
    else:
        # If no mask provided, treat all pixels as valid
        valid_mask = torch.ones((inputs.shape[0], 1, inputs.shape[2]), device=inputs.device, dtype=inputs.dtype)

    # We sum across dim=2 (the flattened pixels) to get totals PER CLASS
    # Invalid pixels are excluded via multiplication by valid_mask
    TP = (inputs * targets * valid_mask).sum(dim=2)    
    FP = ((1 - targets) * inputs * valid_mask).sum(dim=2)
    FN = (targets * (1 - inputs) * valid_mask).sum(dim=2)
    
    # 4. Calculate the Tversky index (Yields a score for each class, per image)
    tversky_index = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)

    # NaN-safety for focal transform:
    tversky_index = tversky_index.clamp(0.0, 1.0)
    focal_base = (1.0 - tversky_index).clamp(min=epsilon, max=1.0)

    
    # Apply the standard focal-Tversky transform (1 - TI)^gamma before averaging.
    focal_tversky = focal_base.pow(gamma)

    # 5. Average the focal scores across all classes and batches
    return focal_tversky.mean()
    

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
        valid_mask = None
        if ignore_mask is not None:
            valid_mask = ~prepare_binary_valid_mask(ignore_mask, logits)
        confusion = compute_binary_confusion(
            logits,
            masks,
            valid_mask=valid_mask,
            threshold=ACTIVE_BINARY_METRIC_THRESHOLD,
        )
        state["tp"] += confusion["tp"].item()
        state["fp"] += confusion["fp"].item()
        state["fn"] += confusion["fn"].item()
        state["tn"] += confusion["tn"].item()
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


def should_save_checkpoint(epoch, epochs, save_every=50):
    return epoch == epochs or (save_every > 0 and epoch % save_every == 0)


def active_dict_collate(batch):
    if not batch:
        raise ValueError("Empty batch")

    images = torch.stack([b[0] for b in batch], dim=0)
    masks = []
    valid_masks = []
    metas = []

    for image, mask, meta in batch:
        if mask is None:
            raise ValueError("Active training requires labeled samples with masks.")
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f"Active mask must have shape [1,H,W], got {tuple(mask.shape)}")

        vm = meta.get("valid_mask", None)
        if vm is None:
            raise KeyError("Sample metadata is missing 'valid_mask'.")
        if not isinstance(vm, torch.Tensor):
            vm = torch.as_tensor(vm)
        if vm.ndim == 3:
            if vm.shape[0] != 1:
                raise ValueError(f"valid_mask must have shape [H,W] or [1,H,W], got {tuple(vm.shape)}")
            vm = vm.squeeze(0)
        elif vm.ndim != 2:
            raise ValueError(f"valid_mask must have shape [H,W] or [1,H,W], got {tuple(vm.shape)}")

        expected_hw = tuple(image.shape[-2:])
        if tuple(vm.shape) != expected_hw:
            y0 = meta.get("patch_y0", None)
            x0 = meta.get("patch_x0", None)
            ps = meta.get("patch_size", None)
            if y0 is None or x0 is None or ps is None:
                raise ValueError(
                    f"valid_mask shape mismatch without patch metadata: got {tuple(vm.shape)}, "
                    f"expected {expected_hw}"
                )
            vm = vm[y0 : y0 + ps, x0 : x0 + ps]

        if tuple(vm.shape) != expected_hw:
            raise ValueError(
                f"valid_mask shape mismatch after patch alignment: got {tuple(vm.shape)}, "
                f"expected {expected_hw}"
            )

        masks.append(mask)
        valid_masks.append(vm.to(dtype=torch.bool))
        metas.append(dict(meta))

    return {
        'image': images,
        'mask': torch.stack(masks, dim=0),
        'valid_mask': torch.stack(valid_masks, dim=0),
        'meta': metas,
    }


def build_multimodal_pretrain_loader(
    combined_root,
    *,
    batch_size,
    num_workers=0,
    shuffle=False,
):
    dataset = FusedDataset.from_combined_manifest(
        combined_root,
        mode="none",
        return_mode="paired",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_pretrain_collate,
    )
    return dataset, loader


def _validate_multimodal_pretrain_batch(batch):
    required = {"sar", "optical", "valid_mask", "meta"}
    missing = sorted(name for name in required if name not in batch)
    if missing:
        raise KeyError(f"Multimodal pretrain batch is missing keys: {missing}")

    sar = batch["sar"]
    optical = batch["optical"]
    valid_mask = batch["valid_mask"]
    metas = batch["meta"]

    if sar.ndim != 4:
        raise ValueError(f"Expected batch['sar'] with shape [B,C,H,W], got {tuple(sar.shape)}")
    if optical.ndim != 4:
        raise ValueError(
            f"Expected batch['optical'] with shape [B,C,H,W], got {tuple(optical.shape)}"
        )
    if valid_mask.ndim not in {3, 4}:
        raise ValueError(
            "Expected batch['valid_mask'] with shape [B,H,W] or [B,1,H,W], "
            f"got {tuple(valid_mask.shape)}"
        )
    if sar.shape[0] != optical.shape[0] or sar.shape[0] != valid_mask.shape[0]:
        raise ValueError("Multimodal batch tensors disagree on batch dimension")
    if tuple(sar.shape[-2:]) != tuple(optical.shape[-2:]):
        raise ValueError("SAR and optical spatial dims differ in multimodal batch")
    valid_hw = tuple(valid_mask.shape[-2:])
    if valid_hw != tuple(sar.shape[-2:]):
        raise ValueError(
            "valid_mask spatial dims do not match multimodal tensors: "
            f"valid_mask={valid_hw}, image={tuple(sar.shape[-2:])}"
        )
    if not isinstance(metas, list) or len(metas) != sar.shape[0]:
        raise ValueError("batch['meta'] must be a list aligned to batch size")
    for idx, meta in enumerate(metas):
        if not isinstance(meta, dict):
            raise TypeError(f"batch['meta'][{idx}] must be a dict")
        forbidden = [name for name in ("mask_path", "label_path") if name in meta]
        if forbidden:
            raise ValueError(
                "Multimodal pretraining batch must not carry label-bearing metadata; "
                f"found {forbidden} at meta index {idx}"
            )

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 5e-3,
        gradient_clipping: float = 0.5,
        tv_alpha = 0.4, #confirm default for Tversky alpha
        tv_beta = 0.6, #confirm default for Tversky beta
        tv_gamma= 4.0 / 3.0,
        adam_betas = (0.9, 0.999),
        n_classes = 1
        ):
    """
    Active supported mode: Phase 1 SAR-only binary flood segmentation.
    Multiclass training remains in the repo but is not supported on this path.
    """
    require_active_binary_mode(n_classes, "train_model()")

    #optimizer setup
    optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    betas=adam_betas, 
    weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=8)

    writer = SummaryWriter(log_dir=output_dir / 'logs')
    logging.info(f"Hyperparameters: {locals()}")

    
    #variables to initialize best validation score tracking for checkpointing
    best_val_iou = -1.0
    best_epoch = -1

    grad_scaler = torch.amp.GradScaler(device=device.type, enabled=amp)
    global_step = 0

    #define loss function with Tversky
    criterion = lambda inputs, targets, vm=None: focal_tversky_loss(
        inputs, targets,
        alpha=tv_alpha,
        beta=tv_beta,
        gamma=tv_gamma,
        valid_mask=vm
    )


    for epoch in range(1, epochs + 1):
        #model into training mode
        model.train()
        epoch_loss = 0
        last_val_score = None

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
                true_masks = true_masks.to(device=device)
                valid_mask = valid_mask.to(device=device)

                #forward pass
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    target_masks = prepare_binary_target(true_masks).to(
                        device=device, dtype=masks_pred.dtype
                    )
                    valid_mask = prepare_binary_valid_mask(valid_mask, masks_pred)
                    loss = criterion(masks_pred, target_masks, valid_mask)
                        
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
                if val_loader is not None and progress_bar.n >= n_train:
                    # 1. Log Weights and Gradients Histograms
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            writer.add_histogram(f'Weights/{tag}', value.data.cpu(), global_step)
                        if value.grad is not None:
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                writer.add_histogram(f'Gradients/{tag}', value.grad.data.cpu(), global_step)

                    # 2. Run Evaluation
                    val_score = evaluate(model, val_loader, device, amp, criterion, n_classes=n_classes)
                    last_val_score = val_score

                    # val_score is a dict from evaluate()
                    for k, v in val_score.items():
                        # TensorBoard expects numeric scalars; skip non-scalars defensively
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f'Validation/{k}', v, global_step)
                            
                    
                    scheduler.step(val_score["val_flood_iou"])

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
                        writer.add_image('Visuals/Mask_True', target_masks[0].float().cpu(), global_step)
                        
                        # Visualization uses the active validation threshold but is not reused by metric code.
                        pred_mask = (
                            torch.sigmoid(masks_pred)[0, 0] > ACTIVE_BINARY_METRIC_THRESHOLD
                        ).float().cpu().unsqueeze(0)
                        
                        writer.add_image('Visuals/Mask_Pred', pred_mask, global_step)
                    except Exception as e:
                        logging.warning(f"Could not log to TensorBoard: {e}")

        
        if last_val_score is not None:
            print(
                f"Validation Results:\n"
                f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n"
                f"  Loss:      {last_val_score['val_loss']:.4f}\n"
                f"  Accuracy:  {last_val_score['val_accuracy']:.4f}\n"
                f"  mIoU:      {last_val_score['val_mIoU']:.4f}\n"
                f"  Flood Metrics -> "
                f"IoU: {last_val_score['val_flood_iou']:.4f} | "
                f"Prec: {last_val_score['val_flood_precision']:.4f} | "
                f"Recall: {last_val_score['val_flood_recall']:.4f} | "
                f"F1: {last_val_score['val_flood_f1']:.4f}"
            )
            # Check if this is the best validation score
            current_val_score = last_val_score['val_flood_iou']
            if current_val_score > best_val_iou:
                best_val_iou = current_val_score
                best_epoch = epoch
                # Save best checkpoint
                if save_checkpoint:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict = model.state_dict()
                    state_dict['mask_values'] = dataset.mask_values
                    torch.save(state_dict, str(dir_checkpoint / 'best_checkpoint.pth'))
                    logging.info(f'Best checkpoint saved! (Epoch {epoch}, IoU: {best_val_iou:.4f})')

        else:
            print(
                f"Epoch {epoch}/{epochs} complete:\n"
                f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n"
                f"  Train Loss: {(epoch_loss / max(len(train_loader), 1)):.4f}\n"
                f"  Validation: disabled"
            )
        
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            if should_save_checkpoint(epoch, epochs):
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
                
    # Log best epoch at the end of training
    if last_val_score is not None:
        logging.info(f'Training completed. Best validation IoU: {best_val_iou:.4f} at epoch {best_epoch}')


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
    parser.add_argument(
        '--classes',
        '-c',
        type=int,
        default=1,
        help='Number of classes. Active Phase 1 training supports binary flood segmentation only (use 1).',
    )
    parser.add_argument('--img-dir', type=str, default=None, help='Path to SAR image .tif/.tiff files')
    parser.add_argument('--mask-dir', type=str, default=None, help='Path to mask .tif/.tiff files')
    parser.add_argument(
        '--multimodal-pretrain-root',
        type=str,
        default=None,
        help='Path to Combined or Combined/manifest.csv for strict multimodal pretraining dataloader validation.',
    )
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker count')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='runs/default',help='Directory to save logs and checkpoints')

    return parser.parse_args()



#argparse usage in main
if __name__ == '__main__':
    args = get_args()

    if args.multimodal_pretrain_root:
        dataset, loader = build_multimodal_pretrain_loader(
            args.multimodal_pretrain_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        batch = next(iter(loader))
        _validate_multimodal_pretrain_batch(batch)
        logging.info(
            "Multimodal pretraining batch ready: sar=%s optical=%s valid_mask=%s strict=%s samples=%s",
            tuple(batch["sar"].shape),
            tuple(batch["optical"].shape),
            tuple(batch["valid_mask"].shape),
            dataset.strict_pairing,
            len(batch["meta"]),
        )
        raise SystemExit(0)

    require_active_binary_mode(args.classes, "train.py")
    if not args.img_dir or not args.mask_dir:
        raise ValueError("--img-dir and --mask-dir are required for the SAR training path")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


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
    mask_id_suffix_map = dict(DEFAULT_MASK_ID_SUFFIX_MAP)
    ids = list_ids_from_dir(args.img_dir, recursive=True)
    paired_ids_list = []
    missing_in_masks = 0
    for image_id in ids:
        mask_id = resolve_mask_id(image_id, mask_dir, mask_id_suffix_map)
        if mask_id is not None:
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

    num_workers = getattr(args, 'num_workers', 0)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=active_dict_collate
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=active_dict_collate
        )
    dir_checkpoint = output_dir / 'checkpoints'
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
