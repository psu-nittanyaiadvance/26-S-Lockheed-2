from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval import evaluate  # noqa: E402
from train import finalize_metrics, init_metric_state, update_metric_state  # noqa: E402
from train import should_save_checkpoint  # noqa: E402


class _FixedLogitNet(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits


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


def test_binary_metric_sanity_check_matches_hand_counted_confusion() -> None:
    logits = torch.tensor(
        [
            [[[2.0, 2.0], [-2.0, -2.0]]],
            [[[-2.0, 2.0], [-2.0, -2.0]]],
        ]
    )
    masks = torch.tensor(
        [
            [[[1, 0], [1, 0]]],
            [[[0, 1], [0, 1]]],
        ],
        dtype=torch.uint8,
    )
    ignore_mask = torch.tensor(
        [
            [[[0, 0], [0, 0]]],
            [[[0, 0], [0, 1]]],
        ],
        dtype=torch.bool,
    )

    state = init_metric_state(1)
    update_metric_state(state, logits, masks, 1, ignore_mask=ignore_mask)
    metrics = finalize_metrics(state, 1)

    assert state["tp"] == pytest.approx(2.0)
    assert state["fp"] == pytest.approx(1.0)
    assert state["fn"] == pytest.approx(1.0)
    assert state["tn"] == pytest.approx(3.0)
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["dice"] == pytest.approx(2.0 / 3.0)


def test_evaluate_binary_metrics_use_active_shapes_valid_mask_and_threshold() -> None:
    logits = torch.tensor(
        [
            [[[2.0, 2.0], [-2.0, -2.0]]],
            [[[-2.0, 2.0], [-2.0, -2.0]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "image": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        "mask": torch.tensor(
            [
                [[[1, 0], [1, 0]]],
                [[[0, 1], [0, 1]]],
            ],
            dtype=torch.uint8,
        ),
        "valid_mask": torch.tensor(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 0]],
            ],
            dtype=torch.bool,
        ),
    }

    def criterion(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        return pred.new_tensor(0.0)

    metrics = evaluate(
        _FixedLogitNet(logits),
        [batch],
        device=torch.device("cpu"),
        amp=False,
        criterion=criterion,
        n_classes=1,
    )

    assert metrics["val_accuracy"] == pytest.approx(5.0 / 7.0)
    assert metrics["val_mIoU"] == pytest.approx(0.55)
    assert metrics["val_flood_iou"] == pytest.approx(0.5)
    assert metrics["val_flood_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["val_flood_recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["val_flood_f1"] == pytest.approx(2.0 / 3.0, rel=1e-5)


def test_short_runs_still_save_a_checkpoint_on_final_epoch() -> None:
    assert should_save_checkpoint(35, 35) is True
    assert should_save_checkpoint(50, 100) is True
    assert should_save_checkpoint(35, 100) is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
