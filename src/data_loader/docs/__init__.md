# `__init__.py`

## Purpose

Defines the public import surface for `data_loader`. It re-exports the dataset classes, manifest reader, splitting utilities, statistics utility, collate functions, discovery helpers, validation helper, and transform factories that downstream code is expected to import.

## Why This File Exists

The package has many implementation modules, but training code should not need to know every module path. `__init__.py` provides stable imports such as:

```python
from data_loader import FusedDataset, build_train_transforms, multimodal_pretrain_collate
```

This is used directly by `src/Multi_modal_src/train.py`.

## Dependencies

Internal: `combined_manifest`, `sar_dataset`, `optical_dataset`, `fused_dataset`, `patch_dataset`, `splits`, `stats`, `collate`, `discover_ids`, `validate_dataset`, and `augmentations`.

External: none directly.

## Classes

No classes are defined here.

## Functions

No functions are defined here. The module imports and exposes functions from implementation modules.

## Data Contracts

The `__all__` list is the package-level contract. Names not included in `__all__` may still be importable by module path, but are not part of the declared top-level API.

ENFORCED:

- Python import resolution will fail if any re-exported module or symbol is missing.

ASSUMED:

- Downstream scripts prefer `from data_loader import ...` over deep imports.
- Adding new public loader utilities should include both an import and an `__all__` entry.

## Tensor Shape Expectations

None directly. Shape contracts are defined in the re-exported implementations.

## Metadata Structure

None directly.

## Valid Mask Handling

None directly. `default_collate`, `multimodal_pretrain_collate`, datasets, and transforms own valid-mask semantics.

## Transform / Augmentation Behavior

Only re-exports `build_train_transforms`, `build_val_transforms`, and `Compose`.

## Design Decisions

- The public API intentionally includes both low-level dataset classes and operational utilities such as validation and stats.
- `validate_combined_manifest_samples` is not exported, even though `fused_dataset.py` uses it internally.
- `make_stratified_split` is defined in `splits.py` but not exported.

## Edge Cases and Failure Modes

- Broken imports here break every `from data_loader import ...` consumer.
- Export drift can occur when a new function is added to a module but not re-exported.

## Modification Risks

- Removing names can break training code without touching loader logic.
- Re-exporting private helpers increases external coupling and makes refactoring harder.

## Example Usage

From `src/Multi_modal_src/train.py`:

```python
from data_loader import FusedDataset, build_train_transforms, build_val_transforms, multimodal_pretrain_collate
```

## Open Questions / Ambiguities

- Whether `make_stratified_split` and `validate_combined_manifest_samples` should be public is unclear.
- There is no package-level version or explicit compatibility policy.

