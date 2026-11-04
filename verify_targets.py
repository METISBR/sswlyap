"""Compare regenerated results with the numbers printed in the manuscript.

    python verify_targets.py [--results results] [--tol-abs 0.002] [--tol-rel 0.02]

Checks, per table cell:
  * Table 1 (tab:deriv): FD extra evaluations exactly; FD relative error within
    a factor 2 (it is a floating-point rounding quantity); FD/AD time ratio only
    reported (wall-clock, hardware-dependent) with a loose factor-3 band.
  * Table 2 (tab:dtlz2): mean and sd of Delta_p within max(tol_abs, tol_rel*value);
    mean FE exactly; Wilcoxon marker identical.
  * Table 3 (tab:sensitivity): mean and sd within the same tolerance.
Exit status 0 iff every deterministic and statistical check passes.
"""

import argparse
import json
import sys
from pathlib import Path

from tables_io import read_csv


def close(value: float, target: float, tol_abs: float, tol_rel: float) -> bool:
    return abs(value - target) <= max(tol_abs, tol_rel * abs(target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--targets", type=Path, default=Path(__file__).with_name("targets.json"))
    parser.add_argument("--tol-abs", type=float, default=0.002)
    parser.add_argument("--tol-rel", type=float, default=0.02)
    args = parser.parse_args()
    targets = json.loads(args.targets.read_text())
    failures, checked = [], 0

    def check(label: str, ok: bool, detail: str) -> None:
        nonlocal checked
        checked += 1
        if not ok:
            failures.append(f"FAIL {label}: {detail}")

    deriv = args.results / "table_deriv" / "derivative_scaling.csv"
    if deriv.exists():
        for row in read_csv(deriv):
            t = targets["table_deriv"][row["n_var"]]
            tag = f"deriv n={row['n_var']}"
            check(f"{tag} fd_extra", int(row["fd_extra_function_evaluations"]) == t["fd_extra"],
                  f"{row['fd_extra_function_evaluations']} != {t['fd_extra']}")
            err = float(row["relative_jacobian_error"])
            check(f"{tag} rel_err", t["rel_err"] / 2 <= err <= t["rel_err"] * 2,
                  f"{err:.2e} vs {t['rel_err']:.1e}")
            ratio = float(row["fd_to_ad_time_ratio"])
            print(f"  [info] {tag}: FD/AD time ratio {ratio:.1f} (manuscript {t['time_ratio']})")
            check(f"{tag} time_ratio band", t["time_ratio"] / 3 <= ratio <= t["time_ratio"] * 3,
                  f"{ratio:.1f} vs {t['time_ratio']}")
    else:
        print(f"  [skip] {deriv} not found")

    summary = args.results / "table2" / "summary.csv"
    if summary.exists():
        for row in read_csv(summary):
            t = targets["table_dtlz2"][row["n_obj"]][row["algorithm"]]
            tag = f"table2 m={row['n_obj']} {row['algorithm']}"
            mean, sd = float(row["mean_delta_p"]), float(row["std_delta_p"])
            check(f"{tag} mean", close(mean, t[0], args.tol_abs, args.tol_rel), f"{mean:.4f} vs {t[0]}")
            check(f"{tag} sd", close(sd, t[1], args.tol_abs, args.tol_rel), f"{sd:.4f} vs {t[1]}")
            check(f"{tag} FE", round(float(row["mean_evaluations"])) == t[2],
                  f"{float(row['mean_evaluations']):.0f} vs {t[2]}")
            check(f"{tag} marker", row["vs_ssw_ad"] == t[3], f"{row['vs_ssw_ad']} vs {t[3]}")
    else:
        print(f"  [skip] {summary} not found")

    sens = args.results / "table3" / "summary.csv"
    if sens.exists():
        for row in read_csv(sens):
            t = targets["table_sensitivity"][row["n_obj"]][str(float(row["epsilon"]))]
            tag = f"table3 m={row['n_obj']} eps={row['epsilon']}"
            mean, sd = float(row["mean_delta_p"]), float(row["std_delta_p"])
            check(f"{tag} mean", close(mean, t[0], args.tol_abs, args.tol_rel), f"{mean:.4f} vs {t[0]}")
            check(f"{tag} sd", close(sd, t[1], args.tol_abs, args.tol_rel), f"{sd:.4f} vs {t[1]}")
    else:
        print(f"  [skip] {sens} not found")

    for line in failures:
        print(line)
    print(f"{checked - len(failures)}/{checked} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
