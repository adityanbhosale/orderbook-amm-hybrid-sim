"""
Cross-market weights from factor loadings (cosine similarity).

Ported from:
  /Users/adityabhosale/Downloads/Projects/lmsr-preclinical-markets/sim/agentic.py
  ``cross_weights_from_loadings``
"""
from __future__ import annotations

import numpy as np


def cross_weights_from_loadings(
    loadings: np.ndarray,
    primary_markets: tuple[int, ...],
    observed_markets: tuple[int, ...],
    min_weight: float = 0.1,
) -> dict[tuple[int, int], float]:
    norms = np.linalg.norm(loadings, axis=1)
    out: dict[tuple[int, int], float] = {}
    for i in primary_markets:
        for j in observed_markets:
            if i == j:
                continue
            denom = norms[i] * norms[j]
            if denom < 1e-12:
                continue
            cos_sim = float(loadings[i] @ loadings[j] / denom)
            if abs(cos_sim) >= min_weight:
                out[(i, j)] = cos_sim
    return out
