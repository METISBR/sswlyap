# sswlyap

Code and data for *Lyapunov Stability of Stochastic Vector Optimization: Theory
and Numerical Implementation* (JOTA-D-26-00246). It regenerates the three
numerical tables of the paper: derivative cost (`tab:deriv`), the DTLZ2
comparison (`tab:dtlz2`) and the noise sensitivity (`tab:sensitivity`).

Stock `pymoo` + `JAX`; nothing else is needed.

## Requirements

| | |
|---|---|
| Python | 3.12 |
| Packages | `requirements.txt`: `numpy==2.4.6`, `scipy==1.17.1`, `pymoo==0.6.1.6`, `jax==0.10.2`, `jaxlib==0.10.2`, `joblib==1.6.0` |
| Hardware | CPU only (results below: Apple M5, 10 cores) |
| Time | about 11 min for everything with `--jobs 8` |

Keep `numpy<2.5`: `pymoo 0.6.1.x` calls `np.row_stack`, which NumPy 2.5 removed,
and the reference directions of NSGA-III and Tchebycheff-AD then fail.

```bash
python3.12 -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # or: conda env create -f environment.yml
```

## Run

```bash
python run_table2.py --smoke    # ~15 s: 1 seed, m=3, five algorithms; run this first
python run_table2.py            # Table 2 -> results/table2/
python run_table3.py            # Table 3 -> results/table3/
python run_table_deriv.py       # Table 1 -> results/table_deriv/  (times wall-clock: run on an idle machine)
python verify_targets.py        # regenerated results vs the numbers printed in the paper
python compare_archived.py      # regenerated runs vs results/paper/, seed by seed
```

`run_table2.py` and `run_table3.py` take `--jobs`, `--runs`, `--budget`,
`--population`, `--objectives`, `--output`.

## Protocol

| | |
|---|---|
| Problem | DTLZ2, `m` in {3, 5, 10, 15}, `n = m + 9` variables in `[0,1]^n` |
| Budget, population | 30,000 objective evaluations (strict), 100 |
| Runs | 30 per configuration, seeds `71000 ... 71029`, the same for every algorithm |
| SSW | step `0.05`, noise `epsilon = 0.02`, archive 100 truncated by crowding distance, FD step `1e-6`; FD evaluations are charged to the budget |
| Tchebycheff-AD | augmented Tchebycheff, one energy-uniform weight per point, projected Adam (lr 0.02), AD gradients; each value-and-gradient call is charged as `N` evaluations |
| NSGA-II, NSGA-III | pymoo defaults; NSGA-III with 100 energy-uniform reference directions |
| Indicator | `Delta_1 = max(GD_1, IGD_1)` against a fixed 4,000-point reference set on the positive unit hypersphere |
| Test | paired two-sided Wilcoxon signed-rank, level 0.05, against SSW-AD |
| Sensitivity | SSW-AD, `epsilon` in {0, 0.01, 0.02, 0.05, 0.15}, `m` in {3, 10} |
| Derivative cost | `m = 3`, `N = 100`, `n` in {12, 24, 48, 96, 192}, float64, 10 warmed repetitions, medians |

Tables 2 and 3 use JAX **float32**; the population, pymoo and the metrics are
float64. Table 1 uses JAX float64. Set `SSW_PRECISION=float64` only to run a
different experiment: it does not reproduce the numbers here.

## Data

`results/paper/` is the per-run output the paper tables were computed from
(`all_runs.csv`, `summary.csv`, `sensitivity_all_runs.csv`,
`sensitivity_summary.csv`, `derivative_scaling.csv`, `configuration.json` with
versions and seeds). `results/table2`, `results/table3` and
`results/table_deriv` are what the scripts in this directory produce.

Regenerated against `results/paper/` (`compare_archived.py`):

* SSW-AD, SSW-FD, NSGA-II, NSGA-III (480 runs) and Table 3 (300 runs): `Delta_p`,
  archive size and evaluation count are bit-identical in every run.
* Tchebycheff-AD (120 runs): not bit-identical. Per-seed `Delta_p` differs by at
  most 5.8e-4 (`m=3`), 6.4e-4 (`m=5`), 2.1e-2 (`m=10`), 1.0e-3 (`m=15`). The
  regenerated means are 0.1001, 0.4417, 0.7445, 0.8795 against 0.1000, 0.4416,
  0.7451, 0.8794 printed in the paper; Wilcoxon markers are unchanged. The cause
  was not established.
* Table 1: FD extra evaluations and relative errors are identical. The FD/AD
  time ratio depends on the machine: 1.3, 3.2, 7.2, 16.9, 74.0 here against 1.4,
  4.8, 8.1, 18.1, 57.5 in the paper.

`verify_targets.py` passes 115/115 checks: exact FE and FD evaluation counts,
identical Wilcoxon markers, `Delta_p` mean and sd within `max(0.002, 2%)` of the
paper, FD error within a factor 2 and FD/AD time ratio within a factor 3.

## Files

| | |
|---|---|
| `ssw.py` | SSW as a pymoo `Algorithm`: Jacobian modes `provided`, `autodiff`, `finite_difference`; batched simplex QP; crowding-distance archive; strict budget |
| `tchebycheff_ad.py` | Tchebycheff-AD baseline |
| `problems.py` | DTLZ2 in JAX, checked against pymoo's DTLZ2 before every run |
| `delta_p.py` | `Delta_p`, reference set, non-dominated filter |
| `experiments.py` | one-run entry points; fixes precision and threading |
| `tables_io.py` | CSV output, environment metadata, Wilcoxon summaries |
| `run_table2.py`, `run_table3.py`, `run_table_deriv.py` | the three experiments |
| `verify_targets.py`, `targets.json` | check against the paper |
| `compare_archived.py` | check against `results/paper/` |

## SSW on your own problem

```python
from pymoo.optimize import minimize
from ssw import SSW

algorithm = SSW(n_points=100, step_size=0.05, epsilon=0.02,
                jacobian_mode="autodiff", autodiff_func=my_jax_objective)  # x -> F(x), pointwise
result = minimize(my_pymoo_problem, algorithm, ("n_eval", 30000), seed=1)
```

`jacobian_mode="finite_difference"` needs no JAX but charges `2nN` evaluations
per generation to the budget; `"provided"` takes `jac=...`. Box bounds are
required.

## Limits

Verified on CPU only; GPU back-ends are untested. One benchmark family (DTLZ2),
by design. Exact digits depend on the pinned versions.
