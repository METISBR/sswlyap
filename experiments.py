"""Single-run entry points shared by the table runners.

Importing this module fixes the JAX precision (``SSW_PRECISION``, default
``float32``, the precision of the archived Tables 2 and 3; the derivative-cost
table sets ``float64`` itself) *before* any JAX array is created, and pins every worker to one
BLAS/XLA thread so that ``joblib`` parallelism over runs does not oversubscribe
the CPU.  Each ``run_*`` function returns ``(F, charged_evaluations, seconds)``
for one DTLZ2 run with ``n = m + 9`` variables.
"""

import os

# One thread per worker; parallelism comes from running many seeds at once.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
# NumPy/pymoo parts of every run are float64 regardless of this setting.

import time

import jax
import numpy as np

PRECISION = os.environ.get("SSW_PRECISION", "float32").strip().lower()
if PRECISION not in ("float64", "float32"):
    raise ValueError("SSW_PRECISION must be 'float64' or 'float32'")
jax.config.update("jax_enable_x64", PRECISION == "float64")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from problems import check_against_pymoo, dtlz2_jax, dtlz2_problem
from ssw import SSW
from tchebycheff_ad import run_tchebycheff

ALGORITHMS = ("SSW-AD", "SSW-FD", "Tchebycheff-AD", "NSGA-II", "NSGA-III")

STEP_SIZE = 0.05  # sigma
MAIN_EPSILON = 0.02
FD_STEP = 1e-6
SEED_BASE = 71_000


def seed_plan(runs: int) -> list[int]:
    """Common seed plan: run r of every algorithm uses seed ``SEED_BASE + r``."""
    return [SEED_BASE + index for index in range(runs)]


def run_ssw(
    n_obj: int, seed: int, budget: int, population: int, epsilon: float, derivative: str
) -> tuple[np.ndarray, int, float]:
    """SSW with ``derivative`` in {"autodiff", "finite_difference"}."""
    kwargs = dict(
        n_points=population,
        archive_size=population,
        step_size=STEP_SIZE,
        epsilon=epsilon,
        strict_evaluation_budget=True,
    )
    if derivative == "autodiff":
        kwargs.update(jacobian_mode="autodiff", autodiff_func=dtlz2_jax(n_obj))
    elif derivative == "finite_difference":
        kwargs.update(jacobian_mode="finite_difference", finite_difference_step=FD_STEP)
    else:
        raise ValueError(f"unknown derivative mode: {derivative}")

    started = time.perf_counter()
    result = minimize(
        dtlz2_problem(n_obj), SSW(**kwargs), ("n_eval", budget), seed=seed, verbose=False
    )
    elapsed = time.perf_counter() - started
    return (
        np.asarray(result.F, dtype=float),
        int(result.algorithm.evaluator.n_eval),
        elapsed,
    )


def run_evolutionary(
    name: str, n_obj: int, seed: int, budget: int, population: int
) -> tuple[np.ndarray, int, float]:
    """NSGA-II or NSGA-III (energy-uniform reference directions, fixed seed)."""
    if name == "NSGA-II":
        algorithm = NSGA2(pop_size=population)
    elif name == "NSGA-III":
        directions = get_reference_directions(
            "energy", n_dim=n_obj, n_points=population, seed=73_001 + n_obj
        )
        algorithm = NSGA3(ref_dirs=directions, pop_size=population)
    else:
        raise ValueError(f"unknown evolutionary algorithm: {name}")

    started = time.perf_counter()
    result = minimize(
        dtlz2_problem(n_obj), algorithm, ("n_eval", budget), seed=seed, verbose=False
    )
    elapsed = time.perf_counter() - started
    return (
        np.asarray(result.F, dtype=float),
        int(result.algorithm.evaluator.n_eval),
        elapsed,
    )


def run_algorithm(
    name: str, n_obj: int, seed: int, budget: int, population: int, epsilon: float = MAIN_EPSILON
) -> tuple[np.ndarray, int, float]:
    """Dispatch one of ``ALGORITHMS`` (SSW variants use ``epsilon``)."""
    if name == "SSW-AD":
        return run_ssw(n_obj, seed, budget, population, epsilon, "autodiff")
    if name == "SSW-FD":
        return run_ssw(n_obj, seed, budget, population, epsilon, "finite_difference")
    if name == "Tchebycheff-AD":
        return run_tchebycheff(n_obj, seed, budget, population)
    return run_evolutionary(name, n_obj, seed, budget, population)


def check_problem_definitions(objectives) -> None:
    for n_obj in objectives:
        check_against_pymoo(n_obj)
