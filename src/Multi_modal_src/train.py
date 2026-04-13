from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from Multi_modal_src.DeCURLoss import DeCURLoss
from Multi_modal_src.eval import build_decur_batch_views, evaluate_decur
from data_loader import FusedDataset, build_train_transforms, build_val_transforms, multimodal_pretrain_collate
from decur.model import DeCUR

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


MODEL_SAR_CHANNELS = 2
MODEL_OPTICAL_CHANNELS = 13


# Fallback writer so the training loop can keep a single logging code path
# even on machines where TensorBoard is not installed yet.
class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def add_text(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device) -> Any:
    # The batch is a nested dict/list structure, so recurse until every tensor
    # has been moved onto the target accelerator.
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def _require_supported_channel_config(args: argparse.Namespace) -> None:
    if args.sar_channels != MODEL_SAR_CHANNELS or args.optical_channels != MODEL_OPTICAL_CHANNELS:
        raise ValueError(
            "The current DeCUR model supports only "
            f"sar_channels={MODEL_SAR_CHANNELS} and optical_channels={MODEL_OPTICAL_CHANNELS}; "
            f"got sar_channels={args.sar_channels}, optical_channels={args.optical_channels}"
        )


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Optional[Dataset]]:
    _require_supported_channel_config(args)

    # Strict manifest loading guarantees SAR and optical stay paired in the
    # same canonical order before any random train/val split happens.
    fused_dataset = FusedDataset.from_combined_manifest(
        args.combined_root,
        mode="none",
        return_mode="paired",
        sar_dataset_kwargs={
            "log_transform": args.sar_log_transform,
            "expected_img_bands": args.sar_channels,
        },
        optical_dataset_kwargs={
            "expected_img_bands": args.optical_channels,
            "reflectance_clip_percentile": args.optical_clip_percentile,
        },
        require_spatial_match=True,
        validate=args.validate_manifest,
    )

    val_size = int(len(fused_dataset) * args.validation_split)
    train_size = len(fused_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_base, val_base = random_split(fused_dataset, [train_size, val_size], generator=generator)

    return train_base, val_base if val_size > 0 else None


def create_writer(output_dir: Path):
    if SummaryWriter is None:
        logging.warning("TensorBoard is unavailable in this environment; using terminal logging only.")
        return _NullSummaryWriter()
    return SummaryWriter(log_dir=str(output_dir / "logs"))


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    # Save enough state to resume training, not just to reuse model weights.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        str(checkpoint_path),
    )


def train_model(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    train_dataset, val_dataset = build_datasets(args)
    logging.info(
        "Dataset ready: train_samples=%s val_samples=%s",
        len(train_dataset),
        0 if val_dataset is None else len(val_dataset),
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": multimodal_pretrain_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    sar_transform = build_train_transforms(modality="sar")
    opt_transform = build_train_transforms(modality="optical")
    val_sar_transform = build_val_transforms()
    val_opt_transform = build_val_transforms()

    model = DeCUR().to(device=device)
    loss_fn = DeCURLoss(common_dim=args.common_dim, lambda_param=args.lambda_param).to(device=device)

    try:
        import torch_optimizer as t_o
    except ImportError as exc:
        raise ImportError(
            "Multimodal DeCUR training requires the 'torch_optimizer' package for LARS."
        ) from exc

# 1. Isolate parameters for LARS
    regular_params = []
    bias_bn_params = []

    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    bias_bn_params.append(param)
        else:
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if param_name == 'bias':
                    bias_bn_params.append(param)
                else:
                    regular_params.append(param)

    #HARD CODED HYPER PARAMETERS FROM DeCUR paper
    param_groups = [
        {
            'params': regular_params,
            'lr': 0.05,            # Base weights LR
            'weight_decay': 1e-6, # Base weight decay
        },
        {
            'params': bias_bn_params,
            'lr': 0.0012,         # Bias/BN LR
            'weight_decay': 0.0,  # Exclude from weight decay
        }
    ]

    # 3. Initialize LARS
    optimizer = t_o.LARS(param_groups, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler(device="cuda", enabled=args.amp)
    writer = create_writer(output_dir)
    writer.add_text("hparams", json.dumps(vars(args), indent=2), global_step=0)

    best_val_loss = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    autocast_device = device.type if device.type != "mps" else "cpu"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        running_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        
        # 1. TRADITIONAL INNER LOOP (Mathematically correct for DeCUR)
        for batch in progress:
            views = build_decur_batch_views(
                batch,
                sar_transform=sar_transform,
                opt_transform=opt_transform,
            )
            views = move_to_device(views, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=autocast_device, enabled=args.amp):
                # Calculations based on full batch statistics as per Eq. 1 [cite: 79]
                z_sar_1, z_sar_2, z_opt_1, z_opt_2 = model(
                    views["sar_view_1"],
                    views["sar_view_2"],
                    views["opt_view_1"],
                    views["opt_view_2"],
                )
                loss = loss_fn(z_sar_1, z_sar_2, z_opt_1, z_opt_2)

            scaler.scale(loss).backward()
            
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                
            scaler.step(optimizer)
            scaler.update()

            # Logging updates
            current_loss = loss.item()
            running_loss += current_loss
            running_batches += 1
            global_step += 1

            writer.add_scalar("train/loss_step", current_loss, global_step)
            progress.set_postfix(loss=f"{current_loss:.4f}")

        # 2. OUTER LOOP (Epoch-level Metrics & Validation)
        train_loss = running_loss / max(running_batches, 1)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        metrics: Dict[str, float] = {"train_loss": train_loss}

        # The paper uses a Cosine Decay Schedule 
        # Stepping here ensures the decay happens after all batches in the epoch
        scheduler.step() 

        # get_last_lr() returns a list of learning rates corresponding to your param_groups
        current_lrs = scheduler.get_last_lr()
        
        # Log the learning rate for the regular parameters (Index 0)
        writer.add_scalar("train/lr_base", current_lrs[0], epoch)
        
        # Log the learning rate for the Bias/BatchNorm parameters (Index 1)
        writer.add_scalar("train/lr_bias_bn", current_lrs[1], epoch)

        if val_loader is not None:
            # Evaluation follows the same protocol of frozen encoder/fine-tuning checks 
            val_metrics = evaluate_decur(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                amp=args.amp,
                common_dim=args.common_dim,
                sar_transform=val_sar_transform,
                opt_transform=val_opt_transform,
            )
            metrics.update(val_metrics)

            for key, value in metrics.items():
                if key != "train_loss":
                    writer.add_scalar(f"val/{key.removeprefix('val_')}", value, epoch)

            # Checkpointing based on the loss specified in Equation 6 [cite: 131]
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, epoch=epoch, global_step=global_step,
                    best_val_loss=best_val_loss, args=args,
                )

        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch, global_step=global_step,
            best_val_loss=best_val_loss, args=args,
        )

        logging.info(
            "epoch=%s train_loss=%.4f val_loss=%s duration=%.1fs",
            epoch, train_loss,
            f"{metrics.get('val_loss', 0):.4f}" if val_loader else "N/A",
            time.perf_counter() - epoch_start,
        )

    writer.flush()
    writer.close()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DeCUR multimodal pretraining engine")
    parser.add_argument("--combined-root", type=str, required=True, help="Path to Combined dataset root or manifest.csv")
    parser.add_argument("--output-dir", type=str, default="runs/decur_pretrain", help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    #default informed by DeCUR paper
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")

    #Learning rate and weight decay are cloned from the DeCUR paper and are hardcoded in the optimizer definition

    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation fraction in [0,1)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm; <=0 disables clipping")
    parser.add_argument("--log-every-steps", type=int, default=25, help="Terminal logging frequency in steps")

    #default optimized to 87.5% for SAR-optical scenario (2048*0.875=1792)
    parser.add_argument("--common-dim", type=int, default=1792, help="Number of common embedding dimensions")

    parser.add_argument("--lambda-param", type=float, default=0.0051, help="Off-diagonal penalty weight")
    parser.add_argument("--sar-channels", type=int, default=2, help="Expected SAR band count")
    parser.add_argument("--optical-channels", type=int, default=13, help="Expected optical band count")
    parser.add_argument("--optical-clip-percentile", type=float, default=2.0, help="Optical robust clipping percentile")
    parser.add_argument("--sar-log-transform", action="store_true", help="Apply SAR log transform in the dataset loader")
    parser.add_argument("--validate-manifest", action="store_true", help="Validate paired manifest assets before training")
    args = parser.parse_args()

    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError("--validation-split must be in [0,1)")
    if args.common_dim < 1:
        raise ValueError("--common-dim must be >= 1")
    return args


if __name__ == "__main__":
    train_model(get_args())
