# `example_usage.py`

## Purpose

Provides a SAR-focused quickstart for constructing datasets, computing normalization statistics, creating DataLoaders, and applying `valid_mask` in a loss.

## Why This File Exists

It serves as executable documentation for the basic supervised SAR workflow.

## Dependencies

Internal:

- `dataloader.sar_dataset.SARDataset`
- `dataloader.splits.make_split`
- `dataloader.collate.default_collate`
- `dataloader.stats.compute_running_mean_std`

External:

- `pathlib.Path`
- `torch.utils.data.DataLoader`

Note: the import path is `dataloader.*`, while the package elsewhere is `data_loader.*`. That is a likely stale path unless the environment aliases it.

## Classes

No classes are defined.

## Functions

### `main()`

Purpose: demonstrate weak, strong, and inference SAR dataset construction.

Inputs: none directly; paths and IDs are hardcoded placeholders.

Outputs: none returned; prints batch shapes and metadata.

Side effects:

- Would read rasters if placeholder paths are replaced with real paths.
- Creates DataLoader iterators.

Preconditions:

- Placeholder roots must be replaced with real paths.
- Import path must resolve.

Postconditions:

- Shows the expected `default_collate` output tuple.

Where it is used: runnable as a script.

Why it exists: communicates the recommended loss-masking pattern.

Edge cases: as written, placeholder paths are not real, so it is documentation unless edited.

## Data Contracts

Demonstrates tuple sample schema and collated schema:

| Object | Shape | Meaning |
| --- | --- | --- |
| `images` | `[B,C,H,W]` | SAR batch. |
| `masks` | `[B,1,H,W]` | Binary label batch. |
| `valid_masks` | `[B,H,W]` | Loss mask. |
| `metas` | `list[dict]` | Per-sample metadata. |

## Tensor Shape Expectations

The loss comment expects:

- model predictions `[B,1,H,W]`
- masks `[B,1,H,W]`
- valid masks unsqueezed to `[B,1,H,W]`

## Metadata Structure

Prints `metas[0]` after `default_collate`, so `valid_mask` has been removed and returned separately.

## Valid Mask Handling

The script's comment shows the intended supervised loss pattern:

```python
loss_px = criterion(pred, masks.float())
vm = valid_masks.unsqueeze(1)
loss = (loss_px * vm).sum() / vm.sum().clamp(min=1)
```

## Transform / Augmentation Behavior

No transforms are configured in this example.

## Design Decisions

- Compute stats on `normalize_cfg="none"` before constructing normalized datasets.
- Use the same mean/std for train and validation.
- Use `mode="none"` for inference datasets.

## Edge Cases and Failure Modes

- Import path may be wrong (`dataloader` vs `data_loader`).
- Placeholder paths fail unless replaced.
- `weak_val` is constructed but not used.

## Modification Risks

- Copying the stale import path into new code may fail.
- Omitting `valid_mask` from loss can train on imputed invalid pixels.

## Example Usage

Run after adapting paths:

```bash
python -m data_loader.example_usage
```

## Open Questions / Ambiguities

- Whether this example should be updated to use `from data_loader import ...`.

