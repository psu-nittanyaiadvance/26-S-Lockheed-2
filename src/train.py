
#main function
if __name__ == "__main__":
    pass


"""
train_model() paramaters for adam optimizer and tversky loss
adam_betas: tuple = (0.9, 0.999),
tversky_alpha: float = 0.5,  # Controls penalty for False Positives
tversky_beta: float = 0.5,   # Controls penalty for False Negatives
gradient_clipping: float = 1.0,
"""

"""
optimizer set up

optimizer = optim.Adam(
    model.parameters(), 
    lr=learning_rate, 
    betas=adam_betas, 
    weight_decay=weight_decay
)
"""

"""
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

"""


"""
# Inside the training loop batch processing:
with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    masks_pred = model(images)
    
    # Replace old loss calculation with Tversky
    loss = tversky_loss(
        masks_pred, 
        true_masks.float(), 
        alpha=tversky_alpha, 
        beta=tversky_beta
    )
"""

"""
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
"""

"""
argparse usage in main

if __name__ == '__main__':
    args = get_args()

    # examples from args:
    # args.epochs
    # args.batch_size
    # args.lr
    # args.load
    # args.scale
    # args.val
    # args.amp
    # args.bilinear
    # args.classes
"""







