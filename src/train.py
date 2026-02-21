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
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


def tversky_loss(inputs, targets, alpha, beta, epsilon=1e-6):
    # Apply sigmoid/softmax to get probabilities
    inputs = torch.sigmoid(inputs) if inputs.shape[1] == 1 else F.softmax(inputs, dim=1)
    
    # Flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # True Positives, False Positives, False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    
    tversky_index = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
    return 1 - tversky_index



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
        momentum: float = 0.999,
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
                progress_bar.set_postfix(**{'loss (batch)': loss.item()})


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








