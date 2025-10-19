# app/utils/purged_kfold.py
# -*- coding: utf-8 -*-
from typing import Iterator, Tuple
import numpy as np

class PurgedKFold:
    """
    Purged K-Fold s embargem.
    - n_splits: počet foldů
    - embargo: počet vzorků vynechaných po validačním okně na obou stranách
    """
    def __init__(self, n_splits: int = 5, embargo: int = 0):
        assert n_splits >= 2
        self.n_splits = n_splits
        self.embargo = max(0, int(embargo))

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        indices = np.arange(n_samples)
        current = 0
        for k, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            # Purge + embargo
            left  = max(0, start - self.embargo)
            right = min(n_samples, stop + self.embargo)
            mask = np.ones(n_samples, dtype=bool)
            mask[left:right] = False
            train_idx = indices[mask]
            yield train_idx, test_idx
            current = stop
