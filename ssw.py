"""Stochastic Steepest Weights (SSW) for vector optimization, on stock pymoo.

Implements the drift--diffusion method of Schaeffler, Schultz and Weinzierl
(JOTA 114(1), 2002) with the projected Euler--Maruyama step

    x_next = clip(x - step_size * q(x) + epsilon * sqrt(step_size) * N(0, I)).

``q(x)`` is the minimum-norm element of the convex hull of the objective
gradients, obtained from the simplex QP  min_{a in simplex} a^T (J J^T) a  as
``q = J^T a``.  The Jacobian ``J`` comes from one of three sources:

``provided``           caller-supplied analytic Jacobian (``jac``),
``autodiff``           JAX (jacfwd if n_var <= n_obj else jacrev), batched,
``finite_difference``  centered differences, charged to the evaluation budget.

The trajectories are independent.  A separate bounded non-dominated archive
(duplicates removed in objective space, truncated by crowding distance) is the
object reported as the result.  With ``strict_evaluation_budget`` the run stops
*before* a generation whose full cost (including the 2n finite-difference
evaluations per particle) would exceed an ``("n_eval", E)`` termination.

Requires pymoo 0.6.1.x; JAX is needed only for ``jacobian_mode="autodiff"``.
"""

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.optimum import filter_optimum


def _project_to_simplex_batch(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row of ``v`` onto the probability simplex."""
    values = np.asarray(v, dtype=float)
    if values.shape[1] == 1:
        return np.ones_like(values)
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=float)
    positive = ordered - cumulative / indices[np.newaxis, :] > 0.0
    rho = np.maximum(np.sum(positive, axis=1) - 1, 0).astype(int)
    rows = np.arange(values.shape[0])
    theta = cumulative[rows, rho] / (rho + 1.0)
    projected = np.maximum(values - theta[:, np.newaxis], 0.0)
    total = np.sum(projected, axis=1, keepdims=True)
    return projected / np.maximum(total, 1e-16)


def compute_q_batch(
    jacobians: np.ndarray, max_iter: int = 250, tol: float = 1e-10
) -> np.ndarray:
    """Minimum-norm common descent vectors for a batch of Jacobians.

    Solves min_a a^T G a over the simplex (G = J J^T) by projected gradient with
    step 1/||G||_inf, simultaneously for all particles, and returns J^T a.  The
    weights ``a`` need not be unique when gradients are affinely dependent, but
    ``J^T a`` (the projection of the origin onto the convex hull of the
    gradients) is.
    """
    batch = np.asarray(jacobians, dtype=float)
    gram = batch @ np.swapaxes(batch, 1, 2)
    n_points, n_obj, _ = batch.shape
    alpha = np.full((n_points, n_obj), 1.0 / n_obj, dtype=float)
    lipschitz = np.max(np.sum(np.abs(gram), axis=2), axis=1)
    step = 1.0 / np.maximum(lipschitz, 1e-12)

    for _ in range(max_iter):
        previous = alpha
        gradient = np.einsum("bij,bj->bi", gram, alpha)
        alpha = _project_to_simplex_batch(alpha - step[:, np.newaxis] * gradient)
        if float(np.max(np.linalg.norm(alpha - previous, axis=1))) <= tol:
            break
    return np.einsum("bmn,bm->bn", batch, alpha)


# Compiled batched Jacobians, keyed by (objective function, n_var <= n_obj).
# Re-jitting per run costs more than the run itself; the cache changes no value.
_JAX_JACOBIAN_CACHE: dict = {}


def _jax_batch_jacobian(func, n_var: int, n_obj: int):
    import jax

    forward = n_var <= n_obj
    key = (func, forward)
    if key not in _JAX_JACOBIAN_CACHE:
        # Forward mode: one tangent sweep per variable.  Reverse mode: one
        # pullback per objective.  Pick the cheaper side of the full Jacobian.
        transform = jax.jacfwd if forward else jax.jacrev
        _JAX_JACOBIAN_CACHE[key] = jax.jit(jax.vmap(transform(func)))
    return _JAX_JACOBIAN_CACHE[key]


class SSW(Algorithm):
    """Projected Stochastic Steepest Weights optimizer.

    Parameters
    ----------
    n_points
        Number of independent stochastic trajectories.
    step_size
        Euler--Maruyama step size (sigma in the paper).
    epsilon
        Diffusion intensity.
    jac
        Analytic Jacobian.  Either vectorized (``X`` of shape ``(N, n)`` ->
        ``(N, m, n)``) or pointwise (``x`` -> ``(m, n)``).
    jacobian_mode
        ``"auto"`` (``jac`` if given, else ``autodiff_func`` if given, else
        finite differences), ``"provided"``, ``"autodiff"`` or
        ``"finite_difference"``.
    autodiff_func
        JAX-traceable pointwise vector objective, used only in autodiff mode.
    archive_size
        Capacity of the non-dominated archive (defaults to ``n_points``).
    finite_difference_step
        Step ``h`` of the centered differences; perturbed points are clipped to
        the box, which yields one-sided differences at active bounds.
    strict_evaluation_budget
        Stop before a generation that would exceed the ``n_eval`` budget.
    """

    _JACOBIAN_MODES = {"auto", "provided", "autodiff", "finite_difference"}

    def __init__(
        self,
        n_points: int = 100,
        step_size: float = 0.01,
        epsilon: float = 0.15,
        jac=None,
        jacobian_mode: str = "auto",
        autodiff_func=None,
        archive_size: int | None = None,
        finite_difference_step: float = 1e-7,
        strict_evaluation_budget: bool = True,
        output=MultiObjectiveOutput(),
        **kwargs,
    ):
        super().__init__(output=output, **kwargs)

        if int(n_points) < 1:
            raise ValueError("n_points must be positive")
        if float(step_size) <= 0.0:
            raise ValueError("step_size must be positive")
        if float(epsilon) < 0.0:
            raise ValueError("epsilon must be non-negative")
        if archive_size is not None and int(archive_size) < 1:
            raise ValueError("archive_size must be positive")
        if float(finite_difference_step) <= 0.0:
            raise ValueError("finite_difference_step must be positive")

        mode = str(jacobian_mode).strip().lower()
        if mode not in self._JACOBIAN_MODES:
            allowed = ", ".join(sorted(self._JACOBIAN_MODES))
            raise ValueError(f"jacobian_mode must be one of: {allowed}")
        if mode == "provided" and jac is None:
            raise ValueError("jacobian_mode='provided' requires jac")
        if mode == "autodiff" and autodiff_func is None:
            raise ValueError("jacobian_mode='autodiff' requires autodiff_func")

        self.n_points = int(n_points)
        self.step_size = float(step_size)
        self.epsilon = float(epsilon)
        self.jac = jac
        self.jacobian_mode = mode
        self.autodiff_func = autodiff_func
        self.archive_size = int(archive_size or n_points)
        self.finite_difference_step = float(finite_difference_step)
        self.strict_evaluation_budget = bool(strict_evaluation_budget)

        # ``Algorithm.archive`` is reserved by pymoo; keep ours separate.
        self.ssw_archive = None

    # ----- setup / initialization ------------------------------------------

    def _setup(self, problem, **kwargs):
        if int(problem.n_var) < 1 or int(problem.n_obj) < 1:
            raise ValueError("SSW requires at least one variable and objective")

    def _initialize_infill(self):
        if not self.problem.has_bounds():
            raise ValueError("SSW requires box bounds")
        lower = np.asarray(self.problem.xl, dtype=float)
        upper = np.asarray(self.problem.xu, dtype=float)
        random = self.random_state.random((self.n_points, self.problem.n_var))
        return Population.new("X", lower + random * (upper - lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.ssw_archive = self._truncate_archive(
            filter_optimum(infills, least_infeasible=True)
        )

    # ----- derivative acquisition ------------------------------------------

    def _resolve_jacobian_mode(self) -> str:
        if self.jacobian_mode != "auto":
            return self.jacobian_mode
        if self.jac is not None:
            return "provided"
        if self.autodiff_func is not None:
            return "autodiff"
        return "finite_difference"

    def _provided_jacobian(self, X: np.ndarray) -> np.ndarray:
        expected = (len(X), self.problem.n_obj, self.problem.n_var)
        try:
            batch = np.asarray(self.jac(X), dtype=float)
            if batch.shape == expected:
                return batch
        except (TypeError, ValueError):
            pass
        batch = np.asarray([self.jac(x) for x in X], dtype=float)
        if batch.shape != expected:
            raise ValueError(f"jac returned {batch.shape}; expected {expected}")
        return batch

    def _autodiff_jacobian(self, X: np.ndarray) -> np.ndarray:
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                "JAX is required for jacobian_mode='autodiff'; "
                "install it or pass jac explicitly"
            ) from exc
        batch_jacobian = _jax_batch_jacobian(
            self.autodiff_func, int(self.problem.n_var), int(self.problem.n_obj)
        )
        return np.asarray(batch_jacobian(jnp.asarray(X)), dtype=float)

    def _finite_difference_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Centered differences, batched, charged to the evaluator.

        All 2*n*N perturbed points are clipped to the box (so no infeasible
        point is ever evaluated) and evaluated in two batched calls; the
        denominator uses the actual, possibly one-sided, clipped distance.
        """
        n_points, n_var = X.shape
        n_obj = int(self.problem.n_obj)
        h = self.finite_difference_step

        replicated = np.tile(X[:, np.newaxis, :], (1, n_var, 1))
        perturbation = np.eye(n_var) * h
        plus_batch = np.clip(
            replicated + perturbation[np.newaxis, :, :],
            self.problem.xl,
            self.problem.xu,
        )
        minus_batch = np.clip(
            replicated - perturbation[np.newaxis, :, :],
            self.problem.xl,
            self.problem.xu,
        )

        pop_plus = Population.new("X", plus_batch.reshape(-1, n_var))
        pop_minus = Population.new("X", minus_batch.reshape(-1, n_var))
        self.evaluator.eval(self.problem, pop_plus)
        self.evaluator.eval(self.problem, pop_minus)

        f_plus = pop_plus.get("F").reshape(n_points, n_var, n_obj)
        f_minus = pop_minus.get("F").reshape(n_points, n_var, n_obj)
        coordinate = np.arange(n_var)
        denominator = (
            plus_batch[:, coordinate, coordinate]
            - minus_batch[:, coordinate, coordinate]
        )
        if np.any(denominator <= 0.0):
            raise ValueError("finite differences require non-degenerate bounds")
        return ((f_plus - f_minus) / denominator[:, :, np.newaxis]).transpose(0, 2, 1)

    def _compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        mode = self._resolve_jacobian_mode()
        if mode == "provided":
            return self._provided_jacobian(X)
        if mode == "autodiff":
            return self._autodiff_jacobian(X)
        return self._finite_difference_jacobian(X)

    # ----- Euler--Maruyama iteration ---------------------------------------

    def _infill(self):
        X = np.asarray(self.pop.get("X"), dtype=float)
        if self.strict_evaluation_budget and not self._generation_fits_budget(len(X)):
            self.termination.terminate()
            return None
        descent = compute_q_batch(self._compute_jacobian(X))
        noise = self.random_state.standard_normal(X.shape)
        next_x = (
            X
            - self.step_size * descent
            + self.epsilon * np.sqrt(self.step_size) * noise
        )
        next_x = np.clip(next_x, self.problem.xl, self.problem.xu)
        return Population.new("X", next_x)

    def _generation_fits_budget(self, population_size: int) -> bool:
        """True if the next generation's full cost fits an ``n_eval`` budget."""
        maximum = int(
            getattr(getattr(self, "termination", None), "n_max_evals", 0) or 0
        )
        if maximum <= 0:
            return True
        derivative_batches = (
            2 * int(self.problem.n_var)
            if self._resolve_jacobian_mode() == "finite_difference"
            else 0
        )
        required = int(population_size) * (1 + derivative_batches)
        used = int(getattr(self.evaluator, "n_eval", 0) or 0)
        return used + required <= maximum

    # ----- archive ---------------------------------------------------------

    def _truncate_archive(self, archive):
        if archive is None or len(archive) == 0:
            return archive
        objectives = np.asarray(archive.get("F"), dtype=float)
        _, unique_indices = np.unique(objectives, axis=0, return_index=True)
        archive = archive[np.sort(unique_indices)]
        if len(archive) <= self.archive_size:
            return archive
        crowding = calc_crowding_distance(np.asarray(archive.get("F"), dtype=float))
        order = np.argsort(-crowding, kind="mergesort")
        return archive[order[: self.archive_size]]

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            return
        self.pop = infills
        candidates = (
            infills
            if self.ssw_archive is None
            else Population.merge(self.ssw_archive, infills)
        )
        self.ssw_archive = self._truncate_archive(
            filter_optimum(candidates, least_infeasible=True)
        )

    def _set_optimum(self):
        self.opt = (
            self.ssw_archive
            if self.ssw_archive is not None and len(self.ssw_archive) > 0
            else filter_optimum(self.pop, least_infeasible=True)
        )
