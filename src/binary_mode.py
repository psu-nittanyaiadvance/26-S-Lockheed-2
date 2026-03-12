import torch


ACTIVE_BINARY_METRIC_THRESHOLD = 0.5


def require_active_binary_mode(n_classes, context):
    if n_classes != 1:
        raise NotImplementedError(
            f"{context} only supports the active Phase 1 binary flood path "
            f"(n_classes=1, 1=flood / 0=background)."
        )


def prepare_binary_logits(logits):
    if logits.ndim != 4:
        raise ValueError(f"Binary logits must have shape [B,1,H,W], got {tuple(logits.shape)}")
    if logits.shape[1] != 1:
        raise ValueError(f"Binary logits must have one channel, got {tuple(logits.shape)}")
    return logits


def prepare_binary_target(mask):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Binary target must have shape [B,1,H,W], got {tuple(mask.shape)}")

    if mask.shape[1] != 1:
        raise ValueError(f"Binary target must have one channel, got {tuple(mask.shape)}")

    return mask.float()


def prepare_binary_valid_mask(valid_mask, reference_tensor):
    if valid_mask is None:
        return None

    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    elif valid_mask.ndim != 4:
        raise ValueError(
            f"Binary valid_mask must have shape [B,H,W] or [B,1,H,W], got {tuple(valid_mask.shape)}"
        )

    if valid_mask.shape[1] != 1:
        raise ValueError(f"Binary valid_mask must have one channel, got {tuple(valid_mask.shape)}")

    reference_shape = prepare_binary_logits(reference_tensor).shape
    if tuple(valid_mask.shape) != reference_shape:
        raise ValueError(
            f"Binary valid_mask shape mismatch: got {tuple(valid_mask.shape)}, "
            f"expected {tuple(reference_shape)}"
        )

    return valid_mask.to(device=reference_tensor.device, dtype=torch.bool)


def compute_binary_confusion(logits, target_mask, valid_mask=None, threshold=ACTIVE_BINARY_METRIC_THRESHOLD):
    logits = prepare_binary_logits(logits)
    target_mask = prepare_binary_target(target_mask).to(device=logits.device)
    valid_mask = prepare_binary_valid_mask(valid_mask, logits)

    probs = torch.sigmoid(logits)
    pred_pos = probs > threshold
    true_pos = target_mask > 0.5

    if valid_mask is None:
        valid_mask = torch.ones_like(pred_pos, dtype=torch.bool)

    pred_pos = pred_pos & valid_mask
    true_pos = true_pos & valid_mask
    true_neg = (~true_pos) & valid_mask
    pred_neg = (~pred_pos) & valid_mask

    return {
        "tp": (pred_pos & true_pos).sum(),
        "fp": (pred_pos & true_neg).sum(),
        "fn": (pred_neg & true_pos).sum(),
        "tn": (pred_neg & true_neg).sum(),
    }
