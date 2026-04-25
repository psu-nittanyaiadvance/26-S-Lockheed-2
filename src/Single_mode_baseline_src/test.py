import argparse
import logging
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Imports from your repository structure
from .binary_mode import ACTIVE_BINARY_METRIC_THRESHOLD, prepare_binary_target, prepare_binary_valid_mask, require_active_binary_mode
from .eval import evaluate
from models.UNet.UNetModel import UNet as UNetModel
from data_loader import PatchDataset, SARDataset, list_ids_from_dir, validate_sample_shapes

# We need the same collate function and map used in training
from .train import active_dict_collate, resolve_mask_id, DEFAULT_MASK_ID_SUFFIX_MAP, focal_tversky_loss


def calculate_mask_iou(pred, target, valid_mask=None):
    """Calculates IoU for a single pair of binary prediction and target masks."""
    global global_intersection, global_union

    # Convert to boolean and squeeze to safely remove the channel dimension [1, 256, 256] -> [256, 256]
    p = pred.bool().squeeze()
    t = target.bool().squeeze()
    
    # If a valid mask is provided, squeeze it and filter the arrays to only look at valid pixels
    if valid_mask is not None:
        vm = valid_mask.bool().squeeze()
        p = p[vm]
        t = t[vm]
    
    intersection = (p & t).float().sum().item()
    union = (p | t).float().sum().item()
    
    # If union is 0, both masks are entirely background (a perfect match)
    if union == 0:
        return 1.0 
        
    global_intersection += intersection
    global_union += union

    return intersection / union


def get_test_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained UNet model on the validation set')
    parser.add_argument('--load', '-f', type=str, required=True, help='Path to the .pth checkpoint file')
    parser.add_argument('--img-dir', type=str, required=True, help='Path to SAR image .tif/.tiff files')
    parser.add_argument('--mask-dir', type=str, required=True, help='Path to mask .tif/.tiff files')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (Must match training!)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (Must match training!)')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker count')
    parser.add_argument('--output-dir', type=str, default='test_results', help='Directory to save TensorBoard logs')

    return parser.parse_args()


if __name__ == '__main__':

    global_intersection = 0
    global_union = 0

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_test_args()
    
    require_active_binary_mode(args.classes, "test.py")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 1. Initialize and Load Model
    model = UNetModel(n_channels=2, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    
    logging.info(f'Loading model from {args.load}')
    state_dict = torch.load(args.load, map_location=device)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval() # CRITICAL: Set model to evaluation mode

    # 2. Replicate Data Loading and Splitting Logic
    torch.manual_seed(args.seed)
    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    mask_id_suffix_map = dict(DEFAULT_MASK_ID_SUFFIX_MAP)
    ids = list_ids_from_dir(args.img_dir, recursive=True)
    
    paired_ids_list = []
    for image_id in ids:
        mask_id = resolve_mask_id(image_id, mask_dir, mask_id_suffix_map)
        if mask_id is not None:
            paired_ids_list.append(image_id)

    if len(paired_ids_list) == 0:
        raise ValueError("No paired image/mask IDs found; aborting testing.")

    base_dataset = SARDataset(
        img_root=img_dir,
        mask_root=mask_dir,
        ids_or_paths=paired_ids_list,
        mode='strong',
        mask_id_suffix_map=mask_id_suffix_map,
        expected_img_bands=2,
    )
    base_dataset.mask_values = [0, 1]
    
    # 3. Exact Split Recreation
    val_percent = args.val / 100
    n_val = int(len(base_dataset) * val_percent)
    n_train = len(base_dataset) - n_val
    
    if n_val == 0:
        raise ValueError("Validation percentage is too low or dataset is too small. No validation images found.")

    generator = torch.Generator().manual_seed(args.seed)
    _, val_base = random_split(base_dataset, [n_train, n_val], generator=generator)
    val_set = PatchDataset(val_base, patch_size=256, overlap=0.2)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle during eval
        num_workers=args.num_workers,
        collate_fn=active_dict_collate
    )

    logging.info(f'Validation set loaded with {len(val_set)} patches.')

    # 4. Define Loss (needed for evaluation metrics)
    criterion = lambda inputs, targets, vm=None: focal_tversky_loss(
        inputs, targets,
        alpha=0.4, 
        beta=0.6,
        gamma=4.0 / 3.0,
        valid_mask=vm
    )

    # 5. Run Evaluation
    logging.info('Starting evaluation...')
    with torch.no_grad(): # Disable gradient calculation for speed and memory
        val_score = evaluate(model, val_loader, device, args.amp, criterion, n_classes=args.classes)

    # 6. Log Images directly to TensorBoard
    out_dir = Path(args.output_dir)
    tb_dir = out_dir / 'tensorboard_test_logs'
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(tb_dir))

    logging.info(f"Logging images and IoU to TensorBoard at '{tb_dir.absolute()}'...")

    global_step = 0 

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move images to device
            images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                masks_pred = model(images)
            
            # Convert raw network output into a clean binary mask (0 or 1)
            pred_binary = (torch.sigmoid(masks_pred) > ACTIVE_BINARY_METRIC_THRESHOLD).float()
            
            # Extract the raw valid mask directly from the dataloader batch
            raw_valid_masks = batch.get("valid_mask", None)
            
            # Iterate through the batch
            for i in range(images.shape[0]):
                # Inside the `for i in range(images.shape[0]):` loop, after current_iou is computed
                pred_mask = pred_binary[i].cpu()
                true_mask = batch['mask'][i].float().cpu()

                # Pull the specific valid mask for this image if it exists
                vm = raw_valid_masks[i].cpu() if raw_valid_masks is not None else None

                # Calculate individual mask IoU
                current_iou = calculate_mask_iou(pred_mask, true_mask, valid_mask=vm)
                
                sar_np = images[i][0].cpu().numpy()

                if global_step == 0:
                    print(f"SAR stats — min: {sar_np.min():.4f}, max: {sar_np.max():.4f}, "
                        f"mean: {sar_np.mean():.4f}, std: {sar_np.std():.4f}")

                DISPLAY_MIN, DISPLAY_MAX = -20.0, -2.0
                sar_clipped = np.clip(sar_np, DISPLAY_MIN, DISPLAY_MAX)
                sar_norm = ((sar_clipped - DISPLAY_MIN) / (DISPLAY_MAX - DISPLAY_MIN) * 255).astype(np.uint8)
                Image.fromarray(sar_norm).save(out_dir / f'{global_step:05d}_sar.png')

                gt_np = (true_mask.squeeze().numpy() * 255).astype(np.uint8)
                Image.fromarray(gt_np).save(out_dir / f'{global_step:05d}_gt.png')

                pred_np = (pred_mask.squeeze().numpy() * 255).astype(np.uint8)
                Image.fromarray(pred_np).save(out_dir / f'{global_step:05d}_pred.png')

                # Log Images. Note: images[i][0:1] safely extracts just the first channel (e.g., VV or VH) 
                # to render as grayscale in Tensorboard, avoiding 2-channel errors.
                writer.add_image('Test_Visuals/1_SAR_Image', images[i][0:1].cpu(), global_step)
                writer.add_image('Test_Visuals/2_GroundTruth', true_mask, global_step)
                writer.add_image('Test_Visuals/3_Prediction', pred_mask, global_step)
                
                # Log the IoU to TensorBoard
                writer.add_scalar('Test_Visuals/4_Mask_IoU', current_iou, global_step)
                
                global_step += 1
    
    # 7. Print Results
    print("\n" + "="*40)
    print("FINAL VALIDATION RESULTS")
    print("="*40)
    print(f"  Loss:      {val_score.get('val_loss', 0):.4f}")
    print(f"  Accuracy:  {val_score.get('val_accuracy', 0):.4f}")
    print(f"  mIoU:      {val_score.get('val_mIoU', 0):.4f}")
    print("-" * 40)
    print("  Flood Specific Metrics:")
    print(f"  IoU:       {val_score.get('val_flood_iou', 0):.4f}")
    print(f"  Precision: {val_score.get('val_flood_precision', 0):.4f}")
    print(f"  Recall:    {val_score.get('val_flood_recall', 0):.4f}")
    print(f"  F1 Score:  {val_score.get('val_flood_f1', 0):.4f}")
    print("="*40 + "\n")
    print("\n"*5)
    
    # Safely divide to prevent ZeroDivisionError on entirely dry validation sets
    final_other_iou = (global_intersection / global_union) if global_union > 0 else 1.0
    print(f"  other IoU:       {final_other_iou:.4f}")

    writer.close()
    logging.info("All images and metrics logged to TensorBoard successfully!")
