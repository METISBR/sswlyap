"""CSV output, environment metadata and paired-Wilcoxon summaries."""

import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def environment() -> dict:
    """Versions and hardware of the interpreter that produced the numbers."""
    import jax
    import pymoo
    import scipy

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pymoo": pymoo.__version__,
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_x64_enabled": bool(jax.config.jax_enable_x64),
    }


def write_configuration(path: Path, **protocol) -> None:
    payload = {**environment(), **protocol}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def paired_marker(values: np.ndarray, baseline: np.ndarray, alpha: float = 0.05):
    """Paired two-sided Wilcoxon signed-rank test, runs paired by seed.

    Returns ``(marker, p_value)``: ``"+"`` if significantly better (smaller
    mean) than ``baseline``, ``"-"`` if significantly worse, ``"="`` otherwise.
    """
    if len(values) != len(baseline):
        return "=", 1.0
    try:
        p_value = float(wilcoxon(values, baseline).pvalue)
    except ValueError:  # all paired differences are zero
        return "=", 1.0
    if p_value < alpha:
        return ("+" if np.mean(values) < np.mean(baseline) else "-"), p_value
    return "=", p_value


def summarize_main(rows: list[dict], objectives, algorithms) -> list[dict]:
    """Table 2: mean/sd of Delta_p per (m, algorithm) and Wilcoxon vs SSW-AD."""
    out = []
    for n_obj in objectives:
        sel = lambda a: sorted(
            (r for r in rows if r["n_obj"] == n_obj and r["algorithm"] == a),
            key=lambda r: r["seed"],
        )
        baseline = np.array([r["delta_p"] for r in sel("SSW-AD")])
        for algorithm in algorithms:
            chosen = sel(algorithm)
            values = np.array([r["delta_p"] for r in chosen])
            marker, p_value = ("=", 1.0)
            if algorithm != "SSW-AD":
                marker, p_value = paired_marker(values, baseline)
            out.append(
                {
                    "n_obj": n_obj,
                    "algorithm": algorithm,
                    "mean_delta_p": float(np.mean(values)),
                    "std_delta_p": float(np.std(values, ddof=1)),
                    "mean_gd": float(np.mean([r["gd"] for r in chosen])),
                    "mean_igd": float(np.mean([r["igd"] for r in chosen])),
                    "mean_nondominated": float(
                        np.mean([r["nondominated"] for r in chosen])
                    ),
                    "mean_evaluations": float(
                        np.mean([r["evaluations"] for r in chosen])
                    ),
                    "mean_time_s": float(np.mean([r["time_s"] for r in chosen])),
                    "vs_ssw_ad": marker,
                    "p_value": p_value,
                }
            )
    return out


def summarize_sensitivity(rows: list[dict], objectives, epsilons) -> list[dict]:
    """Table 3: mean/sd of Delta_p per (m, epsilon) and Wilcoxon vs epsilon=0."""
    out = []
    for n_obj in objectives:
        sel = lambda e: sorted(
            (r for r in rows if r["n_obj"] == n_obj and r["epsilon"] == e),
            key=lambda r: r["seed"],
        )
        baseline = np.array([r["delta_p"] for r in sel(0.0)])
        for epsilon in epsilons:
            chosen = sel(epsilon)
            values = np.array([r["delta_p"] for r in chosen])
            marker, p_value = ("=", 1.0)
            if epsilon != 0.0:
                marker, p_value = paired_marker(values, baseline)
            out.append(
                {
                    "n_obj": n_obj,
                    "epsilon": epsilon,
                    "mean_delta_p": float(np.mean(values)),
                    "std_delta_p": float(np.std(values, ddof=1)),
                    "mean_gd": float(np.mean([r["gd"] for r in chosen])),
                    "mean_igd": float(np.mean([r["igd"] for r in chosen])),
                    "mean_time_s": float(np.mean([r["time_s"] for r in chosen])),
                    "vs_epsilon_zero": marker,
                    "p_value": p_value,
                }
            )
    return out
