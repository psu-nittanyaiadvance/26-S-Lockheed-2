from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from Multi_modal_src.DeCURLoss import DeCURLoss
from data_loader import FusedDataset, PatchDataset, build_train_transforms, build_val_transforms
from eval import evaluate_decur
from models.decur.model import DeCUR

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


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


def _as_hw_mask(valid_mask: Optional[torch.Tensor], hw: torch.Size) -> torch.Tensor:
    # Normalise all valid-mask variants to a plain [H, W] boolean tensor.
    if valid_mask is None:
        return torch.ones(hw, dtype=torch.bool)
    if not isinstance(valid_mask, torch.Tensor):
        valid_mask = torch.as_tensor(valid_mask)
    if valid_mask.ndim == 3:
        if valid_mask.shape[0] != 1:
            raise ValueError(f"valid_mask must have shape [H,W] or [1,H,W], got {tuple(valid_mask.shape)}")
        valid_mask = valid_mask.squeeze(0)
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape [H,W], got {tuple(valid_mask.shape)}")
    if tuple(valid_mask.shape) != tuple(hw):
        raise ValueError(f"valid_mask shape mismatch: got {tuple(valid_mask.shape)}, expected {tuple(hw)}")
    return valid_mask.to(dtype=torch.bool)


def _mask_invalid_pixels(image: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    # Invalid pixels are zeroed before they hit the encoder so corrupted
    # regions do not leak signal into the embedding space.
    return image * valid_mask.unsqueeze(0).to(dtype=image.dtype)


class PairedDeCURDataset(Dataset):
    def __init__(self, base_dataset: Dataset, *, sar_channels: int = 2, train: bool = True) -> None:
        self.base_dataset = base_dataset
        self.sar_channels = sar_channels
        self.sar_transform = build_train_transforms(modality="sar") if train else build_val_transforms()
        self.opt_transform = build_train_transforms(modality="optical") if train else build_val_transforms()

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fused_image, _mask, meta = self.base_dataset[idx]
        sample_id = str(meta.get("id", idx))
        sar_channels = int(meta.get("n_sar_bands", self.sar_channels))
        optical_channels = int(meta.get("n_optical_bands", fused_image.shape[0] - sar_channels))
        if fused_image.shape[0] != sar_channels + optical_channels:
            raise ValueError(
                f"Channel split mismatch for id='{sample_id}': "
                f"image has {fused_image.shape[0]} channels, expected {sar_channels + optical_channels}"
            )

        base_valid_mask = _as_hw_mask(meta.get("valid_mask"), fused_image.shape[-2:])

        # FusedDataset returns one concatenated tensor. Split it back into the
        # modality-specific views expected by the DeCUR encoders.
        sar_image = fused_image[:sar_channels].to(dtype=torch.float32)
        opt_image = fused_image[sar_channels : sar_channels + optical_channels].to(dtype=torch.float32)

        sar_meta = {"id": sample_id, "valid_mask": base_valid_mask.clone(), "modality": "sar"}
        opt_meta = {"id": sample_id, "valid_mask": base_valid_mask.clone(), "modality": "optical"}

        # DeCUR needs two augmented views per modality:
        # SAR view 1 / SAR view 2 / optical view 1 / optical view 2.
        sar_view_1, _, sar_meta_1 = self.sar_transform(sar_image.clone(), None, dict(sar_meta))
        sar_view_2, _, sar_meta_2 = self.sar_transform(sar_image.clone(), None, dict(sar_meta))
        opt_view_1, _, opt_meta_1 = self.opt_transform(opt_image.clone(), None, dict(opt_meta))
        opt_view_2, _, opt_meta_2 = self.opt_transform(opt_image.clone(), None, dict(opt_meta))

        sar_valid_1 = _as_hw_mask(sar_meta_1.get("valid_mask"), sar_view_1.shape[-2:])
        sar_valid_2 = _as_hw_mask(sar_meta_2.get("valid_mask"), sar_view_2.shape[-2:])
        opt_valid_1 = _as_hw_mask(opt_meta_1.get("valid_mask"), opt_view_1.shape[-2:])
        opt_valid_2 = _as_hw_mask(opt_meta_2.get("valid_mask"), opt_view_2.shape[-2:])

        return {
            "sample_id": sample_id,
            "sar_view_1": _mask_invalid_pixels(sar_view_1, sar_valid_1),
            "sar_view_2": _mask_invalid_pixels(sar_view_2, sar_valid_2),
            "opt_view_1": _mask_invalid_pixels(opt_view_1, opt_valid_1),
            "opt_view_2": _mask_invalid_pixels(opt_view_2, opt_valid_2),
            "sar_valid_1": sar_valid_1,
            "sar_valid_2": sar_valid_2,
            "opt_valid_1": opt_valid_1,
            "opt_valid_2": opt_valid_2,
        }


def decur_collate(batch) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")
    # Keep the collate output explicit so the train loop can address each
    # modality/view directly without unpacking tuples by position.
    tensor_keys = [
        "sar_view_1",
        "sar_view_2",
        "opt_view_1",
        "opt_view_2",
        "sar_valid_1",
        "sar_valid_2",
        "opt_valid_1",
        "opt_valid_2",
    ]
    collated = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    collated["sample_id"] = [item["sample_id"] for item in batch]
    return collated


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Optional[Dataset]]:
    # Strict manifest loading guarantees SAR and optical stay paired in the
    # same canonical order before any random train/val split happens.
    fused_dataset = FusedDataset.from_combined_manifest(
        args.combined_root,
        mode="none",
        sar_dataset_kwargs={
            "log_transform": args.sar_log_transform,
            "expected_img_bands": args.sar_channels,
            "validate": False,
        },
        optical_dataset_kwargs={
            "expected_img_bands": args.optical_channels,
            "reflectance_clip_percentile": args.optical_clip_percentile,
            "validate": False,
        },
        require_spatial_match=True,
        validate=args.validate_manifest,
    )

    val_size = int(len(fused_dataset) * args.validation_split)
    train_size = len(fused_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_base, val_base = random_split(fused_dataset, [train_size, val_size], generator=generator)

    train_source: Dataset = train_base
    val_source: Optional[Dataset] = val_base if val_size > 0 else None

    if args.patch_size > 0:
        # Patch after the split so patches from the same parent tile do not
        # leak across train and validation.
        train_source = PatchDataset(train_source, patch_size=args.patch_size, overlap=args.patch_overlap)
        if val_source is not None:
            val_source = PatchDataset(val_source, patch_size=args.patch_size, overlap=args.patch_overlap)

    train_dataset = PairedDeCURDataset(train_source, sar_channels=args.sar_channels, train=True)
    val_dataset = None if val_source is None else PairedDeCURDataset(val_source, sar_channels=args.sar_channels, train=False)
    return train_dataset, val_dataset


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
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
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
        "Dataset ready: train_samples=%s val_samples=%s patch_size=%s",
        len(train_dataset),
        0 if val_dataset is None else len(val_dataset),
        args.patch_size,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": decur_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model_args = SimpleNamespace(dim_common=args.common_dim, lambd=args.lambda_param, batch_size=args.batch_size)
    model = DeCUR(model_args).to(device=device)
    loss_fn = DeCURLoss(common_dim=args.common_dim, lambda_param=args.lambda_param).to(device=device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
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
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, device)

            # Forward pass:
            # 1. encode/project all four modality views
            # 2. compute one scalar DeCUR loss for the batch
            # 3. divide by grad_accum_steps so accumulation matches the
            #    effective large-batch objective.
            with torch.autocast(device_type=autocast_device, enabled=args.amp):
                z_sar_1, z_sar_2, z_opt_1, z_opt_2 = model(
                    batch["sar_view_1"],
                    batch["sar_view_2"],
                    batch["opt_view_1"],
                    batch["opt_view_2"],
                )
                raw_loss = loss_fn(z_sar_1, z_sar_2, z_opt_1, z_opt_2)
                loss = raw_loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if batch_index % args.grad_accum_steps == 0 or batch_index == len(train_loader):
                # Optimizer step happens only after the requested number of
                # micro-batches have contributed gradients.
                scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(raw_loss.item())
            running_batches += 1
            global_step += 1

            writer.add_scalar("train/loss_step", float(raw_loss.item()), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/sar_embedding_norm", float(z_sar_1.norm(dim=1).mean().item()), global_step)
            writer.add_scalar("train/opt_embedding_norm", float(z_opt_1.norm(dim=1).mean().item()), global_step)
            progress.set_postfix(loss=f"{raw_loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                logging.info(
                    "epoch=%s step=%s train_loss=%.4f lr=%.3e",
                    epoch,
                    global_step,
                    raw_loss.item(),
                    optimizer.param_groups[0]["lr"],
                )

        train_loss = running_loss / max(running_batches, 1)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        metrics: Dict[str, float] = {"train_loss": train_loss}

        if val_loader is not None:
            # Validation pauses weight updates and measures whether the learned
            # embeddings are aligning across modalities and across views.
            metrics.update(
                evaluate_decur(
                    model=model,
                    dataloader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    amp=args.amp,
                    common_dim=args.common_dim,
                )
            )
            scheduler.step(metrics["val_loss"])
            for key, value in metrics.items():
                if key != "train_loss":
                    writer.add_scalar(f"val/{key.removeprefix('val_')}", value, epoch)

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                # "best.pt" tracks the strongest validation run; "last.pt"
                # below always tracks the most recent training state.
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    args=args,
                )
        else:
            scheduler.step(train_loss)

        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            args=args,
        )

        logging.info(
            "epoch=%s train_loss=%.4f val_loss=%s cross_modal_common=%s duration=%.1fs",
            epoch,
            train_loss,
            "n/a" if "val_loss" not in metrics else f"{metrics['val_loss']:.4f}",
            "n/a" if "val_cross_modal_cosine_common" not in metrics else f"{metrics['val_cross_modal_cosine_common']:.4f}",
            time.perf_counter() - epoch_start,
        )

    writer.flush()
    writer.close()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DeCUR multimodal pretraining engine")
    parser.add_argument("--combined-root", type=str, required=True, help="Path to Combined dataset root or manifest.csv")
    parser.add_argument("--output-dir", type=str, default="runs/decur_pretrain", help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation fraction in [0,1)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm; <=0 disables clipping")
    parser.add_argument("--log-every-steps", type=int, default=25, help="Terminal logging frequency in steps")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5, help="ReduceLROnPlateau decay factor")
    parser.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--common-dim", type=int, default=4096, help="Number of common embedding dimensions")
    parser.add_argument("--lambda-param", type=float, default=0.0051, help="Off-diagonal penalty weight")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size; <=0 uses full tiles")
    parser.add_argument("--patch-overlap", type=float, default=0.2, help="Patch overlap fraction in [0,1)")
    parser.add_argument("--sar-channels", type=int, default=2, help="Expected SAR band count")
    parser.add_argument("--optical-channels", type=int, default=13, help="Expected optical band count")
    parser.add_argument("--optical-clip-percentile", type=float, default=2.0, help="Optical robust clipping percentile")
    parser.add_argument("--sar-log-transform", action="store_true", help="Apply SAR log transform in the dataset loader")
    parser.add_argument("--validate-manifest", action="store_true", help="Validate paired manifest assets before training")
    args = parser.parse_args()

    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError("--validation-split must be in [0,1)")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if not 0.0 <= args.patch_overlap < 1.0:
        raise ValueError("--patch-overlap must be in [0,1)")
    if args.common_dim < 1:
        raise ValueError("--common-dim must be >= 1")
    return args


if __name__ == "__main__":
    train_model(get_args())
