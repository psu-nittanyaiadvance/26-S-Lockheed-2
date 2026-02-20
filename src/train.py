from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import (  # noqa: E402
    PatchDataset,
    SARDataset,
    compute_running_mean_std,
    default_collate,
    list_ids_from_dir,
    make_split,
)
from src.models.unet.model import UNet  # noqa: E402


DEFAULT_MASK_SUFFIX_MAP = {
    "_S1Weak": "_S1OtsuLabelWeak",
    "_S1Hand": "_LabelHand",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net on SAR flood data.")

    parser.add_argument("--mode", choices=["weak", "strong"], default="strong")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-betas", type=str, default="0.9,0.999")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--bilinear", action="store_true", default=False)
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--loss", choices=["tversky", "bce", "ce"], default="tversky")
    parser.add_argument("--tversky-alpha", type=float, default=0.5)
    parser.add_argument("--tversky-beta", type=float, default=0.5)

    parser.add_argument("--use-patches", action="store_true", default=False)
    parser.add_argument("--no-patches", action="store_false", dest="use_patches")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.2)

    parser.add_argument("--use-time-matched", action="store_true", default=False)
    parser.add_argument("--time-matched-root", type=Path, default=Path("data/derived/gee_time_matched"))
    parser.add_argument("--time-matched-manifest", type=Path, default=Path("data/derived/gee_time_matched_manifest.csv"))
    parser.add_argument("--time-matched-missing-policy", choices=["zeros", "skip", "raise"], default="zeros")

    parser.add_argument("--log-transform", action="store_true", default=True)
    parser.add_argument("--no-log-transform", action="store_false", dest="log_transform")
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--max-stats-samples", type=int, default=None)
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)

    parser.add_argument("--img-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--mask-suffix-map", type=str, default=None)
    parser.add_argument("--debug-batch", action="store_true", default=False)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/unet"))
    parser.add_argument("--save-every", type=int, default=1)

    args = parser.parse_args()

    if args.img_root is None:
        if args.mode == "strong":
            args.img_root = Path("datasets/FilteredSouthAsia/HandLabeled/S1Hand")
        else:
            args.img_root = Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak")
    if args.mask_root is None:
        if args.mode == "strong":
            args.mask_root = Path("datasets/FilteredSouthAsia/HandLabeled/LabelHand")
        else:
            args.mask_root = Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak")

    return args


def parse_suffix_map(text: str | None) -> Dict[str, str]:
    if text is None:
        return dict(DEFAULT_MASK_SUFFIX_MAP)
    text = text.strip()
    if not text:
        return {}
    out: Dict[str, str] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid mask suffix map entry: {part}")
        left, right = part.split("=", 1)
        if not left or not right:
            raise ValueError(f"Invalid mask suffix map entry: {part}")
        out[left] = right
    return out


def apply_suffix_map(sample_id: str, suffix_map: Dict[str, str]) -> str:
    for img_suffix, mask_suffix in suffix_map.items():
        if sample_id.endswith(img_suffix):
            return f"{sample_id[:-len(img_suffix)]}{mask_suffix}"
    return sample_id


def resolve_raster_path(root: Path, name_or_id: str) -> Path:
    name = Path(name_or_id).name
    if name.lower().endswith((".tif", ".tiff")):
        return root / name
    tif = root / f"{name_or_id}.tif"
    tiff = root / f"{name_or_id}.tiff"
    if tif.exists():
        return tif
    if tiff.exists():
        return tiff
    return tif


def clean_nonfinite(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    metadata: Dict,
) -> Tuple[torch.Tensor, torch.Tensor | None, Dict]:
    image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image, mask, metadata


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
    ignore_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.shape[1] != 1:
        raise ValueError("Tversky loss currently supports binary segmentation only (classes=1).")
    probs = torch.sigmoid(logits)
    targets = targets.float()

    if ignore_mask is not None:
        valid = ~ignore_mask.bool()
        probs = probs[valid]
        targets = targets[valid]
        if probs.numel() == 0:
            return logits.sum() * 0.0

    probs = probs.view(-1)
    targets = targets.view(-1)

    tp = (probs * targets).sum()
    fp = ((1 - targets) * probs).sum()
    fn = (targets * (1 - probs)).sum()

    tversky_index = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky_index


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def make_autocast(device_type: str, enabled: bool):
    try:
        return autocast(device_type=device_type, enabled=enabled)
    except TypeError:
        return autocast(enabled=enabled)


def make_grad_scaler(device_type: str, enabled: bool) -> GradScaler:
    try:
        return GradScaler(device_type=device_type, enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def compute_running_mean_std_time_matched(
    dataset: torch.utils.data.Dataset,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    n = 0
    mean = None
    M2 = None

    for i in range(total):
        _, _, meta = dataset[i]
        tm = meta.get("time_matched", None)
        if tm is None:
            raise RuntimeError("time_matched is missing; cannot compute stats.")
        if isinstance(tm, torch.Tensor):
            arr = tm.detach().cpu().numpy()
        else:
            arr = np.asarray(tm)

        if arr.ndim != 3:
            raise ValueError(f"Expected time_matched with shape [C,H,W], got {arr.shape}")

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        c = arr.shape[0]
        batch_n = arr.shape[1] * arr.shape[2]
        flat = arr.reshape(c, -1)
        batch_mean = flat.mean(axis=1)
        batch_var = flat.var(axis=1)

        if mean is None:
            mean = batch_mean
            M2 = batch_var * batch_n
            n = batch_n
        else:
            delta = batch_mean - mean
            total_n = n + batch_n
            mean = mean + delta * (batch_n / total_n)
            M2 = M2 + batch_var * batch_n + (delta ** 2) * (n * batch_n / total_n)
            n = total_n

    if mean is None or M2 is None or n == 0:
        raise ValueError("No samples found to compute time-matched statistics")

    var = M2 / n
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def build_ids(img_root: Path, mask_root: Path, suffix_map: Dict[str, str]) -> List[str]:
    image_ids = list_ids_from_dir(img_root)
    ids: List[str] = []
    for sample_id in image_ids:
        mask_id = apply_suffix_map(sample_id, suffix_map)
        mask_path = resolve_raster_path(mask_root, mask_id)
        if mask_path.exists():
            ids.append(sample_id)
    return ids


def build_datasets(
    args: argparse.Namespace,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Optional[Tuple[np.ndarray, np.ndarray]]]:
    suffix_map = parse_suffix_map(args.mask_suffix_map)
    ids = build_ids(args.img_root, args.mask_root, suffix_map)
    if not ids:
        raise RuntimeError("No paired IDs found. Check roots or suffix map.")
    if args.max_ids is not None:
        ids = ids[: args.max_ids]

    train_ids, val_ids = make_split(ids, val_frac=args.val_frac, seed=args.seed)

    raw_train = SARDataset(
        img_root=args.img_root,
        mask_root=args.mask_root,
        ids_or_paths=train_ids,
        mode=args.mode,
        normalize_cfg="none",
        log_transform=args.log_transform,
        transforms=clean_nonfinite,
        mask_id_suffix_map=suffix_map or None,
        use_time_matched=args.use_time_matched,
        time_matched_root=args.time_matched_root,
        time_matched_manifest=args.time_matched_manifest,
        time_matched_missing_policy=args.time_matched_missing_policy,
    )

    tm_mean: Optional[np.ndarray] = None
    tm_std: Optional[np.ndarray] = None
    if args.norm_stats is not None and args.norm_stats.exists():
        stats = np.load(args.norm_stats)
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)
        if args.use_time_matched:
            if "tm_mean" in stats and "tm_std" in stats:
                tm_mean = stats["tm_mean"].astype(np.float32)
                tm_std = stats["tm_std"].astype(np.float32)
            else:
                tm_mean, tm_std = compute_running_mean_std_time_matched(
                    raw_train, max_samples=args.max_stats_samples
                )
                np.savez(args.norm_stats, mean=mean, std=std, tm_mean=tm_mean, tm_std=tm_std)
    else:
        mean, std = compute_running_mean_std(
            raw_train,
            max_samples=args.max_stats_samples,
            require_no_transforms=False,
            include_log_transform_flag=args.log_transform,
        )
        if args.use_time_matched:
            tm_mean, tm_std = compute_running_mean_std_time_matched(
                raw_train, max_samples=args.max_stats_samples
            )
        if args.norm_stats is not None:
            args.norm_stats.parent.mkdir(parents=True, exist_ok=True)
            if args.use_time_matched and tm_mean is not None and tm_std is not None:
                np.savez(args.norm_stats, mean=mean, std=std, tm_mean=tm_mean, tm_std=tm_std)
            else:
                np.savez(args.norm_stats, mean=mean, std=std)

    train_ds = SARDataset(
        img_root=args.img_root,
        mask_root=args.mask_root,
        ids_or_paths=train_ids,
        mode=args.mode,
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
        log_transform=args.log_transform,
        transforms=clean_nonfinite,
        mask_id_suffix_map=suffix_map or None,
        use_time_matched=args.use_time_matched,
        time_matched_root=args.time_matched_root,
        time_matched_manifest=args.time_matched_manifest,
        time_matched_missing_policy=args.time_matched_missing_policy,
    )

    val_ds = SARDataset(
        img_root=args.img_root,
        mask_root=args.mask_root,
        ids_or_paths=val_ids,
        mode=args.mode,
        normalize_cfg={"type": "zscore", "mean": mean, "std": std},
        log_transform=args.log_transform,
        transforms=clean_nonfinite,
        mask_id_suffix_map=suffix_map or None,
        use_time_matched=args.use_time_matched,
        time_matched_root=args.time_matched_root,
        time_matched_manifest=args.time_matched_manifest,
        time_matched_missing_policy=args.time_matched_missing_policy,
    )

    if args.use_patches:
        train_ds = PatchDataset(train_ds, patch_size=args.patch_size, overlap=args.overlap)
        val_ds = PatchDataset(val_ds, patch_size=args.patch_size, overlap=args.overlap)

    tm_stats = (tm_mean, tm_std) if args.use_time_matched else None
    return train_ds, val_ds, tm_stats


def concat_time_matched(
    images: torch.Tensor,
    metas: List[Dict],
    tm_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if not metas:
        return images
    h, w = images.shape[2], images.shape[3]
    tm_list: List[torch.Tensor] = []
    for meta in metas:
        tm = meta.get("time_matched", None)
        if tm is None:
            raise RuntimeError("use_time_matched=True but metadata has no 'time_matched'.")
        if tm.shape[1] != h or tm.shape[2] != w:
            y0 = meta.get("patch_y0")
            x0 = meta.get("patch_x0")
            if y0 is None or x0 is None:
                raise RuntimeError("Time-matched stack shape mismatch and no patch coords.")
            tm = tm[:, y0 : y0 + h, x0 : x0 + w]
        tm_list.append(tm)
    tm_batch = torch.stack(tm_list, dim=0).to(images.device)
    tm_batch = torch.nan_to_num(tm_batch, nan=0.0, posinf=0.0, neginf=0.0)
    if tm_norm is not None:
        tm_mean, tm_std = tm_norm
        tm_batch = (tm_batch - tm_mean) / tm_std
    return torch.cat([images, tm_batch], dim=1)


def init_metric_state(num_classes: int) -> Dict:
    if num_classes == 1:
        return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "valid_batches": 0, "skipped_batches": 0}
    return {
        "tp": torch.zeros(num_classes, dtype=torch.float64),
        "fp": torch.zeros(num_classes, dtype=torch.float64),
        "fn": torch.zeros(num_classes, dtype=torch.float64),
    }


def update_metric_state(
    state: Dict,
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_mask: Optional[torch.Tensor] = None,
) -> None:
    if num_classes == 1:
        preds = torch.sigmoid(logits) > 0.5
        targets = masks > 0.5
        if ignore_mask is not None:
            valid = ~ignore_mask.bool()
            preds = preds[valid]
            targets = targets[valid]
        tp = (preds & targets).sum().item()
        fp = (preds & (~targets)).sum().item()
        fn = ((~preds) & targets).sum().item()
        if tp + fp + fn == 0:
            state["skipped_batches"] += 1
            return
        state["tp"] += tp
        state["fp"] += fp
        state["fn"] += fn
        state["valid_batches"] += 1
        return

    if masks.ndim == 4:
        targets = masks.squeeze(1).long()
    else:
        targets = masks.long()
    preds = torch.argmax(logits, dim=1)

    for c in range(num_classes):
        pred_c = preds == c
        tgt_c = targets == c
        tp = (pred_c & tgt_c).sum().item()
        fp = (pred_c & (~tgt_c)).sum().item()
        fn = ((~pred_c) & tgt_c).sum().item()
        state["tp"][c] += tp
        state["fp"][c] += fp
        state["fn"][c] += fn


def finalize_metrics(state: Dict, num_classes: int) -> Dict[str, float]:
    eps = 1e-6
    if num_classes == 1:
        if state.get("valid_batches", 0) == 0:
            return {"iou": float("nan"), "dice": float("nan")}
        tp = state["tp"]
        fp = state["fp"]
        fn = state["fn"]
        iou = tp / (tp + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        return {"iou": float(iou), "dice": float(dice)}

    tp = state["tp"].numpy()
    fp = state["fp"].numpy()
    fn = state["fn"].numpy()
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return {"iou": float(iou.mean()), "dice": float(dice.mean())}


def _tensor_basic_stats(tensor: torch.Tensor) -> Dict[str, float]:
    t = tensor.detach()
    if not t.is_floating_point():
        t = t.float()
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
    }


def _maybe_channel_stats(tensor: torch.Tensor) -> Optional[str]:
    if tensor.ndim != 4:
        return None
    b, c, _, _ = tensor.shape
    if c > 8:
        return None
    t = tensor.detach()
    if not t.is_floating_point():
        t = t.float()
    means = t.mean(dim=(0, 2, 3))
    stds = t.std(dim=(0, 2, 3))
    mean_str = ", ".join(f"{v:.4f}" for v in means.tolist())
    std_str = ", ".join(f"{v:.4f}" for v in stds.tolist())
    return f"per_channel_mean=[{mean_str}] per_channel_std=[{std_str}]"


def _debug_batch(
    tag: str,
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    args: argparse.Namespace,
    ignore_mask: Optional[torch.Tensor] = None,
) -> None:
    if args.classes == 1:
        metrics_path = "binary(sigmoid+0.5) targets=mask>0.5"
    else:
        metrics_path = "multiclass(argmax) targets=int labels"
    print(f"[debug-batch:{tag}] metrics_path={metrics_path}")
    img_stats = _tensor_basic_stats(images)
    img_chan = _maybe_channel_stats(images)
    print(
        "[debug-batch:image] "
        f"dtype={images.dtype} shape={tuple(images.shape)} "
        f"min={img_stats['min']:.6f} max={img_stats['max']:.6f} "
        f"mean={img_stats['mean']:.6f} std={img_stats['std']:.6f}"
    )
    if img_chan:
        print(f"[debug-batch:image] {img_chan}")

    uniq = torch.unique(masks.detach().cpu())
    uniq_list = uniq.tolist()[:20]
    mask_stats = _tensor_basic_stats(masks)
    print(
        "[debug-batch:mask] "
        f"dtype={masks.dtype} shape={tuple(masks.shape)} "
        f"unique_first20={uniq_list} min={mask_stats['min']:.6f} "
        f"max={mask_stats['max']:.6f} mean={mask_stats['mean']:.6f}"
    )

    logits_stats = _tensor_basic_stats(logits)
    print(
        "[debug-batch:logits] "
        f"dtype={logits.dtype} shape={tuple(logits.shape)} "
        f"min={logits_stats['min']:.6f} max={logits_stats['max']:.6f} "
        f"mean={logits_stats['mean']:.6f}"
    )

    if args.classes == 1:
        probs = torch.sigmoid(logits)
        probs_stats = _tensor_basic_stats(probs)
        print(
            "[debug-batch:probs] "
            f"min={probs_stats['min']:.6f} max={probs_stats['max']:.6f} "
            f"mean={probs_stats['mean']:.6f}"
        )

        preds = probs > 0.5
        targets = masks > 0.5
        if ignore_mask is not None:
            valid = ~ignore_mask.bool()
            preds = preds[valid]
            targets = targets[valid]
            ignore_frac = float(ignore_mask.float().mean().item())
            print(f"[debug-batch:mask] ignore_fraction={ignore_frac:.6f}")

        pred_pos_frac = float(preds.float().mean().item()) if preds.numel() else float("nan")
        tgt_pos_frac = float(targets.float().mean().item()) if targets.numel() else float("nan")
        print(
            "[debug-batch:rates] "
            f"pred_pos_frac={pred_pos_frac:.6f} target_pos_frac={tgt_pos_frac:.6f}"
        )

        tp = int((preds & targets).sum().item()) if preds.numel() else 0
        fp = int((preds & (~targets)).sum().item()) if preds.numel() else 0
        fn = int(((~preds) & targets).sum().item()) if preds.numel() else 0
        union = tp + fp + fn
        print(
            "[debug-batch:counts] "
            f"tp={tp} fp={fp} fn={fn} union={union}"
        )


def _extract_ignore_mask(
    metas: List[Dict],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not metas or not all("ignore_mask" in m for m in metas):
        return None
    masks: List[torch.Tensor] = []
    for meta in metas:
        mask = meta.get("ignore_mask")
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        masks.append(mask)
    ignore = torch.stack(masks, dim=0).to(device, non_blocking=True)
    return ignore


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    tm_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    metric_state = init_metric_state(args.classes)
    debug_printed = False

    for images, masks, metas in loader:
        if masks is None:
            raise RuntimeError("Supervised training requires masks, but got None.")

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        ignore_mask = _extract_ignore_mask(metas, device)

        if args.use_time_matched:
            images = concat_time_matched(images, metas, tm_norm).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with make_autocast(device.type, enabled=args.amp):
            logits = model(images)
            if args.loss == "tversky":
                loss = tversky_loss(
                    logits,
                    masks,
                    args.tversky_alpha,
                    args.tversky_beta,
                    ignore_mask=ignore_mask,
                )
            elif args.loss == "bce":
                if ignore_mask is None:
                    loss = F.binary_cross_entropy_with_logits(logits, masks.float())
                else:
                    per_pixel = F.binary_cross_entropy_with_logits(
                        logits, masks.float(), reduction="none"
                    )
                    valid = ~ignore_mask.bool()
                    if valid.any():
                        loss = per_pixel[valid].mean()
                    else:
                        loss = logits.sum() * 0.0
            else:
                targets = masks.squeeze(1).long()
                loss = F.cross_entropy(logits, targets)

        if args.debug_batch and not debug_printed:
            _debug_batch("train", images, masks, logits, args, ignore_mask=ignore_mask)
            debug_printed = True

        scaler.scale(loss).backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu().item())
        update_metric_state(
            metric_state,
            logits.detach(),
            masks.detach(),
            args.classes,
            ignore_mask=ignore_mask,
        )
        n_batches += 1
        if args.max_batches is not None and n_batches >= args.max_batches:
            break

    metrics = finalize_metrics(metric_state, args.classes)
    return total_loss / max(n_batches, 1), metrics


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    tm_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metric_state = init_metric_state(args.classes)
    debug_printed = False

    for images, masks, metas in loader:
        if masks is None:
            raise RuntimeError("Validation requires masks, but got None.")

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        ignore_mask = _extract_ignore_mask(metas, device)

        if args.use_time_matched:
            images = concat_time_matched(images, metas, tm_norm).to(device, non_blocking=True)

        logits = model(images)
        if args.loss == "tversky":
            loss = tversky_loss(
                logits,
                masks,
                args.tversky_alpha,
                args.tversky_beta,
                ignore_mask=ignore_mask,
            )
        elif args.loss == "bce":
            if ignore_mask is None:
                loss = F.binary_cross_entropy_with_logits(logits, masks.float())
            else:
                per_pixel = F.binary_cross_entropy_with_logits(
                    logits, masks.float(), reduction="none"
                )
                valid = ~ignore_mask.bool()
                if valid.any():
                    loss = per_pixel[valid].mean()
                else:
                    loss = logits.sum() * 0.0
        else:
            targets = masks.squeeze(1).long()
            loss = F.cross_entropy(logits, targets)

        if args.debug_batch and not debug_printed:
            _debug_batch("val", images, masks, logits, args, ignore_mask=ignore_mask)
            debug_printed = True

        total_loss += float(loss.detach().cpu().item())
        update_metric_state(
            metric_state,
            logits.detach(),
            masks.detach(),
            args.classes,
            ignore_mask=ignore_mask,
        )
        n_batches += 1
        if args.max_batches is not None and n_batches >= args.max_batches:
            break

    metrics = finalize_metrics(metric_state, args.classes)
    return total_loss / max(n_batches, 1), metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.loss == "tversky" and args.classes != 1:
        raise ValueError("Tversky loss is configured for binary segmentation only (classes=1).")
    if args.loss == "bce" and args.classes != 1:
        raise ValueError("BCE loss expects classes=1. Use a multiclass loss if needed.")
    if args.loss == "ce" and args.classes <= 1:
        raise ValueError("Cross-entropy loss expects classes > 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    args.amp = amp_enabled
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, tm_stats = build_datasets(args)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=default_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=default_collate,
    )

    sample_img, _, sample_meta = train_ds[0]
    n_channels = int(sample_img.shape[0])
    if args.use_time_matched:
        tm = sample_meta.get("time_matched", None)
        if tm is None:
            raise RuntimeError("use_time_matched=True but sample metadata has no time_matched.")
        n_channels += int(tm.shape[0])

    model = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=tuple(float(x) for x in args.adam_betas.split(",")),
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(device.type, enabled=amp_enabled)

    config_path = args.output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(vars(args)), f, indent=2)

    tm_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if args.use_time_matched:
        if tm_stats is None or tm_stats[0] is None or tm_stats[1] is None:
            raise RuntimeError("Time-matched normalization stats are missing.")
        tm_mean, tm_std = tm_stats
        tm_mean_t = torch.tensor(tm_mean, device=device).view(1, -1, 1, 1)
        tm_std_t = torch.tensor(tm_std, device=device).view(1, -1, 1, 1)
        tm_norm = (tm_mean_t, tm_std_t)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, args, tm_norm
        )
        val_loss, val_metrics = eval_epoch(model, val_loader, device, args, tm_norm)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_iou={train_metrics['iou']:.4f} | train_dice={train_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | val_dice={val_metrics['dice']:.4f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = args.output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = args.output_dir / "checkpoint_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_path,
            )


if __name__ == "__main__":
    main()
