from __future__ import annotations

import random
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np


def make_split(ids: Sequence[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Deterministically split IDs into train/val sets.

    The split is based on a stable sorted ordering of IDs, then a seeded shuffle.
    """
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError(f"val_frac must be in [0, 1], got {val_frac}")

    ids_sorted = sorted(ids)
    ids_shuffled = list(ids_sorted)
    rng = random.Random(seed)
    rng.shuffle(ids_shuffled)

    n_val = int(len(ids_shuffled) * val_frac)
    if val_frac > 0.0 and n_val == 0 and len(ids_shuffled) > 0:
        n_val = 1

    val_ids = ids_shuffled[:n_val]
    train_ids = ids_shuffled[n_val:]
    return train_ids, val_ids


def make_stratified_split(
    ids: Sequence[str],
    val_frac: float,
    seed: int,
    labels: Mapping[str, Union[int, float]],
    bins: int = 5,
) -> Tuple[List[str], List[str]]:
    """
    Stratified split of IDs into train/val sets.

    If labels are floats, values are binned into quantiles before stratification.
    """
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError(f"val_frac must be in [0, 1], got {val_frac}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    ids_sorted = sorted(ids)
    if not ids_sorted:
        return [], []

    try:
        values = [labels[i] for i in ids_sorted]
    except KeyError as exc:
        raise ValueError(f"Missing label for id '{exc.args[0]}'") from exc

    is_float = any(
        isinstance(v, (float, np.floating)) and not isinstance(v, (int, np.integer, bool))
        for v in values
    )

    if is_float:
        arr = np.asarray(values, dtype=np.float64)
        unique_vals = np.unique(arr)
        if unique_vals.size <= 1:
            bin_ids = np.zeros_like(arr, dtype=int)
        else:
            bins_used = min(bins, int(unique_vals.size))
            quantiles = np.linspace(0.0, 1.0, bins_used + 1)
            edges = np.quantile(arr, quantiles)
            edges = np.unique(edges)
            if edges.size <= 1:
                bin_ids = np.zeros_like(arr, dtype=int)
            else:
                bin_ids = np.digitize(arr, edges[1:-1], right=True)
        strata = bin_ids.tolist()
    else:
        strata = [int(v) for v in values]

    strata_to_ids: Dict[int, List[str]] = {}
    for sample_id, stratum in zip(ids_sorted, strata):
        strata_to_ids.setdefault(int(stratum), []).append(sample_id)

    rng = random.Random(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []

    for stratum in sorted(strata_to_ids.keys()):
        stratum_ids = list(strata_to_ids[stratum])
        rng.shuffle(stratum_ids)
        n_val = int(len(stratum_ids) * val_frac)
        if val_frac > 0.0 and n_val == 0 and len(stratum_ids) > 0:
            n_val = 1
        val_ids.extend(stratum_ids[:n_val])
        train_ids.extend(stratum_ids[n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids
