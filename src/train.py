

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


