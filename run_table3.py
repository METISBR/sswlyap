"""Table 3: noise-intensity sensitivity of SSW-AD on DTLZ2.

epsilon in {0, 0.01, 0.02, 0.05, 0.15} at m in {3, 10}; every other setting as
in Table 2 (30 paired runs, 30,000 evaluations, population 100, sigma = 0.05).
epsilon = 0 removes the diffusion term (deterministic multistart descent).

    python run_table3.py --output results/table3
    python run_table3.py --smoke
"""

import argparse
from pathlib import Path

from joblib import Parallel, delayed

import experiments as ex
from delta_p import metrics, reference_front
from tables_io import summarize_sensitivity, write_configuration, write_csv


def one_run(n_obj, epsilon, run_id, seed, budget, population):
    F, evaluations, elapsed = ex.run_ssw(n_obj, seed, budget, population, epsilon, "autodiff")
    return {
        "n_obj": n_obj,
        "n_var": n_obj + 9,
        "epsilon": epsilon,
        "run": run_id,
        "seed": seed,
        "evaluations": evaluations,
        "time_s": elapsed,
        **metrics(F, reference_front(n_obj)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("results/table3"))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--budget", type=int, default=30_000)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--objectives", default="3,10")
    parser.add_argument("--epsilons", default="0,0.01,0.02,0.05,0.15")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true", help="1 seed, m=3, eps in {0, 0.02}")
    args = parser.parse_args()

    if args.smoke:
        args.runs, args.objectives, args.epsilons = 1, "3", "0,0.02"
        if args.output == Path("results/table3"):
            args.output = Path("results/smoke_table3")
    objectives = [int(v) for v in args.objectives.split(",")]
    epsilons = [float(v) for v in args.epsilons.split(",")]
    seeds = ex.seed_plan(args.runs)
    ex.check_problem_definitions(objectives)

    tasks = [
        (m, e, run_id, seed)
        for m in objectives
        for e in epsilons
        for run_id, seed in enumerate(seeds, start=1)
    ]
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_run)(m, e, r, s, args.budget, args.population) for m, e, r, s in tasks
    )

    write_configuration(
        args.output / "configuration.json",
        runs=args.runs,
        budget=args.budget,
        population=args.population,
        objectives=objectives,
        epsilons=epsilons,
        ssw_step_size=ex.STEP_SIZE,
        reference_front_points=4000,
        seed_plan=seeds,
    )
    write_csv(args.output / "all_runs.csv", rows)
    summary = summarize_sensitivity(rows, objectives, epsilons)
    write_csv(args.output / "summary.csv", summary)
    print(f"{'m':>3} {'epsilon':>8} {'mean':>8} {'sd':>8}  vs eps=0")
    for r in summary:
        print(
            f"{r['n_obj']:>3} {r['epsilon']:8.3f} {r['mean_delta_p']:8.4f} "
            f"{r['std_delta_p']:8.4f}  {r['vs_epsilon_zero']}"
        )
    print(f"wrote {args.output}/")


if __name__ == "__main__":
    main()
