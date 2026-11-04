"""Per-seed comparison of the regenerated runs with the data behind the paper.

    python compare_archived.py

``results/paper/`` holds the per-run CSVs the paper tables were computed from.
For every (m, algorithm[, epsilon], seed) this reports whether Delta_p is
bit-identical, the largest absolute difference, and whether the archive size and
the number of evaluations agree.
"""

import argparse
import collections
from pathlib import Path

from tables_io import read_csv


def compare(mine_path, paper_path, group_keys, label):
    key = lambda r: tuple(
        (float(r[k]) if k == "epsilon" else r[k]) for k in group_keys + ("seed",)
    )
    mine = {key(r): r for r in read_csv(mine_path)}
    stats = collections.defaultdict(lambda: [0, 0, 0.0, 0, 0])
    for k, a in ((key(r), r) for r in read_csv(paper_path)):
        r = mine[k]
        s = stats[k[:-1]]
        diff = abs(float(r["delta_p"]) - float(a["delta_p"]))
        s[0] += 1
        s[1] += diff < 1e-12
        s[2] = max(s[2], diff)
        s[3] += r["nondominated"] == a["nondominated"]
        s[4] += r["evaluations"] == a["evaluations"]
    print(f"== {label}")
    print(f"  {'group':<28}{'seeds':>6}{'bit-identical':>15}{'max|diff|':>12}{'same size':>11}{'same FE':>9}")
    for group in sorted(stats, key=str):
        n, same, worst, size, fe = stats[group]
        print(f"  {str(group):<28}{n:>6}{same:>15}{worst:>12.2e}{size:>11}{fe:>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    paper = args.results / "paper"
    compare(args.results / "table2/all_runs.csv", paper / "all_runs.csv",
            ("n_obj", "algorithm"), "Table 2 (tab:dtlz2)")
    compare(args.results / "table3/all_runs.csv", paper / "sensitivity_all_runs.csv",
            ("n_obj", "epsilon"), "Table 3 (tab:sensitivity)")


if __name__ == "__main__":
    main()
