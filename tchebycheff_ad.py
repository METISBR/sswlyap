"""Tchebycheff-AD: gradient-based augmented Tchebycheff scalarization baseline.

One fixed weight vector from an energy-uniform set of reference directions is
assigned to each of the N points.  Each point minimizes its own augmented
Tchebycheff scalarization

    s_w(x) = max_i w_i f_i(x) + 1e-3 * sum_i w_i f_i(x)

by projected Adam (lr 0.02, betas 0.9/0.999, eps 1e-8, clip to [0, 1]^n), with
gradients from JAX automatic differentiation.  A batched value-and-gradient
call is charged as one objective evaluation per point, exactly as for SSW-AD,
and the final objective evaluation of the population is charged as well.
"""

from functools import lru_cache
import time

import jax
import jax.numpy as jnp
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

from problems import dtlz2_jax

LEARNING_RATE = 0.02
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
AUGMENTATION = 1e-3  # weight of the sum term in the augmented Tchebycheff


def tchebycheff_budget_plan(budget: int, population: int) -> tuple[int, int]:
    """Adam updates and charged evaluations that fit a strict budget.

    The initial population is represented by the final batched objective
    evaluation; each Adam update consumes one population-sized batch through
    the value-and-gradient call.  Both are charged, so the total never exceeds
    ``budget``.
    """
    if population < 1:
        raise ValueError("population must be positive")
    if budget < population:
        raise ValueError("budget must accommodate at least one population")
    iterations = (budget - population) // population
    return iterations, (iterations + 1) * population


@lru_cache(maxsize=None)
def _weights(n_obj: int, population: int) -> np.ndarray:
    weights = get_reference_directions(
        "energy", n_dim=n_obj, n_points=population, seed=81_001 + n_obj
    )
    weights = np.maximum(weights, 1e-4)
    return weights / np.sum(weights, axis=1, keepdims=True)


@lru_cache(maxsize=None)
def _compiled(n_obj: int):
    objective = dtlz2_jax(n_obj)

    def scalar_loss(x, weight):
        weighted = weight * objective(x)
        return jnp.max(weighted) + AUGMENTATION * jnp.sum(weighted)

    value_and_gradient = jax.jit(
        jax.vmap(jax.value_and_grad(scalar_loss), in_axes=(0, 0))
    )
    return value_and_gradient, jax.jit(jax.vmap(objective))


def run_tchebycheff(
    n_obj: int, seed: int, budget: int, population: int
) -> tuple[np.ndarray, int, float]:
    """Return ``(F, charged_evaluations, seconds)`` for one run on DTLZ2."""
    value_and_gradient, batch_objective = _compiled(n_obj)
    W = jnp.asarray(_weights(n_obj, population))
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.random((population, n_obj + 9)))
    first_moment = jnp.zeros_like(X)
    second_moment = jnp.zeros_like(X)
    iterations, charged = tchebycheff_budget_plan(budget, population)

    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        _, gradient = value_and_gradient(X, W)
        first_moment = BETA1 * first_moment + (1 - BETA1) * gradient
        second_moment = BETA2 * second_moment + (1 - BETA2) * gradient**2
        corrected_first = first_moment / (1.0 - BETA1**iteration)
        corrected_second = second_moment / (1.0 - BETA2**iteration)
        X = jnp.clip(
            X - LEARNING_RATE * corrected_first / (jnp.sqrt(corrected_second) + ADAM_EPS),
            0.0,
            1.0,
        )
    F = np.asarray(batch_objective(X).block_until_ready(), dtype=float)
    return F, charged, time.perf_counter() - started
