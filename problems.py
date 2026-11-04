"""DTLZ2 in JAX (the differentiable form of F used by the AD modes).

``dtlz2_jax(m)`` returns the pointwise canonical DTLZ2 objective

    g(x)   = sum_{i=m}^{n} (x_i - 1/2)^2,
    f_1(x) = (1 + g) prod_{j=1}^{m-1} cos(pi x_j / 2),
    f_i(x) = (1 + g) prod_{j=1}^{m-i} cos(pi x_j / 2) * sin(pi x_{m-i+1} / 2),

with n = m + 9 variables in [0, 1]^n (Section 4.1 of the paper).  The returned
function object is cached per ``m`` so that JIT-compiled Jacobians are reused
across runs.  ``check_against_pymoo`` verifies that this definition agrees with
pymoo's NumPy DTLZ2, which is what NSGA-II/III and SSW-FD evaluate.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from pymoo.problems import get_problem


@lru_cache(maxsize=None)
def dtlz2_jax(n_obj: int):
    """Pointwise DTLZ2 objective ``x -> F(x)`` in JAX (cached per ``n_obj``)."""

    def objective(x):
        g = jnp.sum((x[n_obj - 1 :] - 0.5) ** 2)
        values = []
        for i in range(n_obj):
            value = 1.0 + g
            for j in range(n_obj - i - 1):
                value = value * jnp.cos(0.5 * jnp.pi * x[j])
            if i > 0:
                value = value * jnp.sin(0.5 * jnp.pi * x[n_obj - i - 1])
            values.append(value)
        return jnp.stack(values)

    return objective


def dtlz2_problem(n_obj: int):
    """pymoo DTLZ2 with n = m + 9 variables in [0, 1]^n."""
    return get_problem("dtlz2", n_var=n_obj + 9, n_obj=n_obj)


def check_against_pymoo(n_obj: int) -> None:
    """Fail early if the JAX and pymoo definitions of DTLZ2 diverge."""
    import jax

    x = np.random.default_rng(4 + n_obj).random((5, n_obj + 9))
    expected = dtlz2_problem(n_obj).evaluate(x)
    actual = np.asarray(jax.vmap(dtlz2_jax(n_obj))(jnp.asarray(x)))
    if not np.allclose(expected, actual, atol=1e-6):
        raise RuntimeError(f"JAX DTLZ2 (m={n_obj}) does not match pymoo")
