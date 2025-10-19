# app/utils/augment.py
# -*- coding: utf-8 -*-
import numpy as np

def timeseries_mixup(Xa: np.ndarray, ya: np.ndarray, p: float = 0.30, alpha: float = 0.2):
    """
    TSMixup: s pravděpodobností p smíchá dva náhodné vzorky.
    Xa: (N, L, C)  ya: (N, ...) binární/pravděpodobnosti
    """
    if np.random.rand() > p: return Xa, ya
    n = Xa.shape[0]
    idx = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha)
    X = lam * Xa + (1.0 - lam) * Xa[idx]
    y = lam * ya + (1.0 - lam) * ya[idx]
    return X, y

def time_mask(X: np.ndarray, p: float = 0.3, max_width: int = 6):
    """
    Time-mask: náhodně vynuluje krátký úsek v čase (per vzorek).
    """
    if np.random.rand() > p: return X
    N, L, C = X.shape
    w = np.random.randint(1, max_width+1)
    s = np.random.randint(0, L - w + 1)
    X2 = X.copy()
    X2[:, s:s+w, :] = 0.0
    return X2