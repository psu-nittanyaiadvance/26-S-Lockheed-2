# `README.md`

## Purpose

Documents the data loading subsystem at a system level. It is the first file a new engineer or reviewer should read before modifying loader behavior.

## Why This File Exists

The subsystem has hidden coupling across datasets, collate functions, transforms, patching, strict manifests, and multimodal training. A top-level guide is needed to explain the overall data flow and cross-file contracts that individual source files cannot fully capture.

## Dependencies

Internal: references every module in `src/data_loader` and training interactions under `src/Multi_modal_src`.

External: none.

## Classes

No classes are defined.

## Functions

No functions are defined.

## Data Contracts

The README defines the public conceptual contracts:

- segmentation tuple sample schema;
- paired multimodal dict sample schema;
- valid-mask semantics;
- label ownership rules;
- strict manifest-pairing rules;
- augmentation alignment rules.

ENFORCED:

- Not enforced by the README itself. Enforcement lives in code modules documented elsewhere.

ASSUMED:

- Developers update the README when changing subsystem-wide behavior.

ACCIDENTAL:

- Documentation can drift from code if future changes are made without updating it.

## Tensor Shape Expectations

The README records canonical shapes:

- segmentation image `[C,H,W]`;
- segmentation mask `[1,H,W]`;
- segmentation collated images `[B,C,H,W]`;
- segmentation collated valid masks `[B,H,W]`;
- paired sample SAR/optical `[C,H,W]`;
- paired collated valid masks `[B,1,H,W]`.

## Metadata Structure

The README documents common metadata keys and how they differ among SAR, optical, fused, paired, and patched samples.

## Valid Mask Handling

The README explicitly states that `valid_mask` is a pixel-quality/loss mask, not a label mask, and that fused valid masks are logical intersections of modality valid masks.

## Transform / Augmentation Behavior

The README describes transform boundaries: geometry must affect image, mask, valid mask, and ignore mask together; radiometric transforms must not alter labels.

## Design Decisions

- Keep system-wide behavior in one place and file-level specifics in `src/data_loader/docs/`.
- Label ENFORCED, ASSUMED, and ACCIDENTAL behavior so reviewers can distinguish guarantees from current implementation quirks.
- Include missing requested files (`src/train.py`, `src/eval.py`) as an explicit trace finding instead of inventing interactions.

## Edge Cases and Failure Modes

- README claims should be verified against code when runtime logic changes.
- Stakeholders may treat documented assumptions as guarantees; assumptions are intentionally labeled.

## Modification Risks

- Replacing the README with a short summary would remove important cross-module context.
- Updating only per-file docs and not README can leave system-level contracts stale.

## Example Usage

Read first, then follow the recommended reading order in the README for implementation details.

## Open Questions / Ambiguities

- The repository does not currently have `src/train.py` or `src/eval.py` at the requested paths, so README interaction notes are limited to existing training/evaluation files.

