"""Table 2: five algorithms on DTLZ2, m in {3, 5, 10, 15}.

Protocol (Section 4.3 of the paper): n = m + 9 variables in [0, 1]^n,
population 100, strict budget of 30,000 objective evaluations, 30 runs with a
common seed plan, Delta_1 against a fixed 4,000-point reference set, paired
two-sided Wilcoxon signed-rank test at 0.05 against SSW-AD.

    python run_table2.py --output results/table2          # full protocol
    python run_table2.py --smoke                          # 1 seed, m = 3
"""

import argparse
from pathlib import Path

from joblib import Parallel, delayed

import experiments as ex
from delta_p import metrics, reference_front
from tables_io import summarize_main, write_configuration, write_csv


def one_run(algorithm, n_obj, run_id, seed, budget, population):
    F, evaluations, elapsed = ex.run_algorithm(algorithm, n_obj, seed, budget, population)
    return {
        "n_obj": n_obj,
        "n_var": n_obj + 9,
        "run": run_id,
        "seed": seed,
        "algorithm": algorithm,
        "evaluations": evaluations,
        "time_s": elapsed,
        **metrics(F, reference_front(n_obj)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("results/table2"))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--budget", type=int, default=30_000)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--objectives", default="3,5,10,15")
    parser.add_argument("--algorithms", default=",".join(ex.ALGORITHMS))
    parser.add_argument("--jobs", type=int, default=-1, help="joblib workers (-1 = all cores)")
    parser.add_argument("--smoke", action="store_true", help="1 seed, m=3 only")
    args = parser.parse_args()

    if args.smoke:
        args.runs, args.objectives = 1, "3"
        if args.output == Path("results/table2"):
            args.output = Path("results/smoke")
    objectives = [int(v) for v in args.objectives.split(",")]
    algorithms = args.algorithms.split(",")
    seeds = ex.seed_plan(args.runs)
    ex.check_problem_definitions(objectives)

    tasks = [
        (algorithm, n_obj, run_id, seed)
        for n_obj in objectives
        for run_id, seed in enumerate(seeds, start=1)
        for algorithm in algorithms
    ]
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_run)(a, m, r, s, args.budget, args.population) for a, m, r, s in tasks
    )

    write_configuration(
        args.output / "configuration.json",
        runs=args.runs,
        budget=args.budget,
        population=args.population,
        objectives=objectives,
        algorithms=algorithms,
        ssw_step_size=ex.STEP_SIZE,
        ssw_main_epsilon=ex.MAIN_EPSILON,
        fd_step=ex.FD_STEP,
        reference_front_points=4000,
        seed_plan=seeds,
    )
    write_csv(args.output / "all_runs.csv", rows)
    if "SSW-AD" in algorithms:
        summary = summarize_main(rows, objectives, algorithms)
        write_csv(args.output / "summary.csv", summary)
        print(f"{'m':>3} {'algorithm':<15} {'mean':>8} {'sd':>8} {'FE':>7}  vs SSW-AD")
        for r in summary:
            print(
                f"{r['n_obj']:>3} {r['algorithm']:<15} {r['mean_delta_p']:8.4f} "
                f"{r['std_delta_p']:8.4f} {r['mean_evaluations']:7.0f}  {r['vs_ssw_ad']}"
            )
    print(f"wrote {args.output}/")


if __name__ == "__main__":
    main()
