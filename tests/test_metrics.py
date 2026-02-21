from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import finalize_metrics, init_metric_state, update_metric_state  # noqa: E402


def test_metrics_applies_sigmoid_for_logits_binary() -> None:
    logits = torch.tensor([[[[0.2, -0.2, 0.6]]]])
    masks = torch.tensor([[[[1, 0, 1]]]], dtype=torch.uint8)

    state = init_metric_state(1)
    update_metric_state(state, logits, masks, 1)
    metrics = finalize_metrics(state, 1)

    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)


def test_union_zero_policy_excludes_empty_batches() -> None:
    logits = torch.full((1, 1, 2, 2), -2.0)
    masks = torch.zeros((1, 1, 2, 2), dtype=torch.uint8)

    state = init_metric_state(1)
    update_metric_state(state, logits, masks, 1)
    metrics = finalize_metrics(state, 1)

    assert math.isnan(metrics["iou"])
    assert math.isnan(metrics["dice"])


def test_ignore_mask_excludes_pixels_from_metrics() -> None:
    logits = torch.tensor([[[[2.0, 2.0]]]])
    masks = torch.tensor([[[[1, 0]]]], dtype=torch.uint8)
    ignore_mask = torch.tensor([[[[0, 1]]]], dtype=torch.bool)

    state = init_metric_state(1)
    update_metric_state(state, logits, masks, 1, ignore_mask=ignore_mask)
    metrics = finalize_metrics(state, 1)

    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
