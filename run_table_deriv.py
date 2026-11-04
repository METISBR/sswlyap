"""Table 1 (``tab:deriv``): cost of acquiring one population's Jacobian.

DTLZ2 with m = 3 objectives, N = 100 particles, n in {12, 24, 48, 96, 192},
float64.  AD = batched ``jacrev`` (n > m), compiled; FD = the 2nN centered-
difference evaluations of the same compiled objective.  Both are timed warm,
serially, as the median of 10 repetitions.  Reported per n: the number of extra
function evaluations FD needs (AD needs none), the FD/AD wall-clock ratio, and
the relative Frobenius error of the FD Jacobian with respect to AD.

Timing is hardware-dependent; run this serially on an otherwise idle machine.
The evaluation counts and relative errors are deterministic.

    python run_table_deriv.py --output results/table_deriv
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("SSW_PRECISION", "float64")

import experiments as ex  # noqa: E402,F401  (fixes precision / threads first)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from problems import dtlz2_jax  # noqa: E402
from tables_io import write_configuration, write_csv  # noqa: E402

N_OBJ = 3
N_VARS = (12, 24, 48, 96, 192)
FD_STEP = 1e-6


def measure(n_var: int, population: int, repetitions: int, rng) -> dict:
    objective = dtlz2_jax(N_OBJ)
    batch_objective = jax.jit(jax.vmap(objective))
    transform = jax.jacrev if n_var > N_OBJ else jax.jacfwd
    batch_jacobian = jax.jit(jax.vmap(transform(objective)))

    X = rng.random((population, n_var))
    X_jax = jnp.asarray(X)
    expected = np.asarray(batch_jacobian(X_jax).block_until_ready())  # warm-up
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        batch_jacobian(X_jax).block_until_ready()
        timings.append(time.perf_counter() - started)
    ad_time = float(np.median(timings))

    repeated = np.tile(X[:, None, :], (1, n_var, 1))
    perturbation = np.eye(n_var) * FD_STEP
    plus = np.clip(repeated + perturbation[None, :, :], 0.0, 1.0)
    minus = np.clip(repeated - perturbation[None, :, :], 0.0, 1.0)
    plus_jax = jnp.asarray(plus.reshape(-1, n_var))
    minus_jax = jnp.asarray(minus.reshape(-1, n_var))
    batch_objective(plus_jax).block_until_ready()  # warm-up
    batch_objective(minus_jax).block_until_ready()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        plus_values = batch_objective(plus_jax).block_until_ready()
        minus_values = batch_objective(minus_jax).block_until_ready()
        timings.append(time.perf_counter() - started)
    fd_time = float(np.median(timings))

    f_plus = np.asarray(plus_values).reshape(population, n_var, N_OBJ)
    f_minus = np.asarray(minus_values).reshape(population, n_var, N_OBJ)
    index = np.arange(n_var)
    denominator = plus[:, index, index] - minus[:, index, index]
    approximate = ((f_plus - f_minus) / denominator[:, :, None]).transpose(0, 2, 1)
    error = float(
        np.linalg.norm(approximate - expected) / max(np.linalg.norm(expected), 1e-16)
    )
    return {
        "n_obj": N_OBJ,
        "n_var": n_var,
        "population": population,
        "ad_time_s": ad_time,
        "fd_time_s": fd_time,
        "fd_to_ad_time_ratio": fd_time / max(ad_time, 1e-16),
        "ad_extra_function_evaluations": 0,
        "fd_extra_function_evaluations": 2 * n_var * population,
        "relative_jacobian_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("results/table_deriv"))
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(1927)
    rows = [measure(n, args.population, args.repetitions, rng) for n in N_VARS]
    write_configuration(
        args.output / "configuration.json",
        n_obj=N_OBJ,
        n_vars=list(N_VARS),
        population=args.population,
        repetitions=args.repetitions,
        fd_step=FD_STEP,
    )
    write_csv(args.output / "derivative_scaling.csv", rows)
    print(f"{'n':>4} {'FD extra evals':>15} {'FD/AD time':>11} {'rel. error':>11}")
    for r in rows:
        print(
            f"{r['n_var']:>4} {r['fd_extra_function_evaluations']:>15,} "
            f"{r['fd_to_ad_time_ratio']:>11.1f} {r['relative_jacobian_error']:>11.2e}"
        )
    print(f"wrote {args.output}/")


if __name__ == "__main__":
    main()
