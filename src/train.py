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
from UNet import UNetModel

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
        adam_betas = (0.9, 0.999) 
        ):

    #optimizer setup
    optimizer = optim.Adam(
    model.parameters(), 
    lr=learning_rate, 
    betas=adam_betas, 
    weight_decay=weight_decay
    )

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
                images, true_masks = batch['image'], batch['mask']

                #check for correct dimensionality
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                #format and load to device
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                #forward pass
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    if model.n_classes == 1:
                        # Add a channel dimension to true_masks so it matches masks_pred shape [Batch, 1, Height, Width]
                        target_masks = true_masks.unsqueeze(1).float()
                        loss = criterion(masks_pred, target_masks)
                    else:
                        # One-hot encode the target masks and rearrange dimensions to match masks_pred
                        target_masks = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                        loss = criterion(masks_pred, target_masks)
                        
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
                if division_step > 0:
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
                        val_score = evaluate(model, val_loader, device, amp, criterion, )

                        logging.info(f'Validation Dice score: {val_score}')

                        # 3. Log Scalars and Images to TensorBoard
                        try:
                            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                            writer.add_scalar('Validation/Dice', val_score, global_step)
                            
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

    return parser.parse_args()



#argparse usage in main
if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetModel(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
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
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
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
            amp=args.amp
        )