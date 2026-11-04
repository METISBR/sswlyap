"""Averaged Hausdorff distance Delta_p (Schuetze et al., IEEE TEVC 16(4), 2012).

    GD_p(A, R)  = ( mean_{a in A} min_{r in R} ||a - r||^p )^(1/p)
    IGD_p(A, R) = ( mean_{r in R} min_{a in A} ||a - r||^p )^(1/p)
    Delta_p     = max(GD_p, IGD_p)

The paper uses p = 1.  ``A`` is the non-dominated part of the returned archive
(``non_dominated``); ``R`` is the fixed reference set of 4,000 points drawn
uniformly on the positive part of the unit hypersphere (``reference_front``),
seeded per objective count so every algorithm sees the same R.
"""

from functools import lru_cache

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial.distance import cdist

REFERENCE_POINTS = 4000


@lru_cache(maxsize=None)
def reference_front(n_obj: int, n_points: int = REFERENCE_POINTS) -> np.ndarray:
    """Fixed Monte Carlo design on the positive unit hypersphere (DTLZ2 front)."""
    rng = np.random.default_rng(910_000 + n_obj)
    front = np.abs(rng.normal(size=(n_points, n_obj)))
    return front / np.linalg.norm(front, axis=1, keepdims=True)


def non_dominated(F: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(F, dtype=float))
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    return values[indices]


def delta_p(A: np.ndarray, R: np.ndarray, p: float = 1.0) -> float:
    """Delta_p(A, R) = max(GD_p, IGD_p)."""
    A = np.atleast_2d(A)
    R = np.atleast_2d(R)
    if len(A) == 0:
        return float("inf")
    distances = cdist(A, R)
    gd = float(np.mean(np.min(distances, axis=1) ** p) ** (1.0 / p))
    igd = float(np.mean(np.min(distances, axis=0) ** p) ** (1.0 / p))
    return max(gd, igd)


def metrics(F: np.ndarray, reference: np.ndarray) -> dict:
    """GD_1, IGD_1, Delta_1 and archive size of the non-dominated part of F."""
    values = non_dominated(F)
    distances = cdist(values, reference)
    gd = float(np.mean(np.min(distances, axis=1)))
    igd = float(np.mean(np.min(distances, axis=0)))
    return {
        "gd": gd,
        "igd": igd,
        "delta_p": max(gd, igd),
        "nondominated": int(len(values)),
    }
