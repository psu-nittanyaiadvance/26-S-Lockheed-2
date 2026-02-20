# Training Guide

This document describes how to run `src/train.py` for strong/weak labels, patching, and time-matched inputs.

## Defaults
1. `--mode strong`
2. `--use-patches` is **OFF** by default. Use `--use-patches` to enable patch training.
3. Loss defaults to `tversky` (binary). Use `--loss ce --classes C` for multi-class.

## Strong Labels (Full-Frame Baseline)
```powershell
python src\train.py --mode strong --no-patches --epochs 20 --batch-size 4 --output-dir runs\unet_strong_full
```

## Strong Labels (Patch Training)
```powershell
python src\train.py --mode strong --use-patches --patch-size 256 --overlap 0.2 --epochs 20 --batch-size 4 --output-dir runs\unet_strong_patches
```

## Weak Labels (Patch Training)
```powershell
python src\train.py --mode weak --use-patches --patch-size 256 --overlap 0.2 --epochs 20 --batch-size 4 --output-dir runs\unet_weak
```

## Multi-Class (Cross-Entropy)
```powershell
python src\train.py --mode strong --loss ce --classes 3 --epochs 20 --batch-size 4 --output-dir runs\unet_strong_ce
```

## Time-Matched (Option B Normalization)
Time-matched channels are normalized separately and concatenated after SAR normalization.
```powershell
python src\train.py --mode strong --use-time-matched --norm-stats runs\norm_stats_tm.npz --epochs 20 --batch-size 4 --output-dir runs\unet_tm
```

## Debug / Smoke Run
```powershell
python src\train.py --mode strong --epochs 1 --batch-size 1 --val-frac 0.2 --max-ids 4 --max-stats-samples 4 --max-batches 1 --output-dir runs\tmp_smoke
```

## Debugging Near-Zero IoU/Dice
Common causes for ~0 IoU/Dice in small debug runs:
1. Logits vs probabilities: metrics use sigmoid + 0.5 threshold for binary. If you inspect logits directly, compare against 0.0 (not 0.5).
2. Empty targets: if a batch has no positive pixels (after ignore), IoU/Dice are excluded. If *all* batches are empty, metrics print as `nan`.
3. Ignore pixels in strong labels: some hand-labeled masks contain `-1` (unlabeled). These pixels are excluded from loss/metrics and reported as `ignore_fraction` in `--debug-batch`.
4. Class imbalance: tiny positive fractions can make early metrics look near-zero even when training is fine.
5. Patches vs full frames: patches change the positive fraction per batch, so metrics can shift sharply.

Recommended debug commands:
```powershell
# Full-frame, single-batch sanity check with batch diagnostics
python src\train.py --mode strong --no-patches --epochs 1 --batch-size 1 --max-batches 1 --debug-batch --output-dir runs\tmp_dbg_full

# Patch-based run (use-patches is OFF by default)
python src\train.py --mode strong --use-patches --epochs 1 --batch-size 1 --max-batches 1 --debug-batch --output-dir runs\tmp_dbg_patches
```

Debug checklist:
1. Confirm `target_pos_frac` is non-zero on at least some batches.
2. Check `ignore_fraction` to understand how much of the mask is excluded.
3. Verify `pred_pos_frac` is reasonable (not all 0 or all 1).

## Notes
1. `--loss tversky` and `--loss bce` require `--classes 1`.
2. `--loss ce` requires `--classes > 1` and expects integer masks `(N,H,W)`; masks from the loader are `(N,1,H,W)` and are squeezed internally.
3. Best checkpoint is saved to `runs/<name>/checkpoint_best.pt` based on lowest validation loss.
