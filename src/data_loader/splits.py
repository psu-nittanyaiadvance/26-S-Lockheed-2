from __future__ import annotations

import random
from typing import List, Sequence, Tuple


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
