"""Run the experiments of the report.

Usage:
    python run_experiments.py --list
    python run_experiments.py --experiment budget              # one experiment
    python run_experiments.py --experiment all --runs 30       # everything
    python run_experiments.py --experiment final               # tuned setting + cross-validation

Each setting of an experiment is run `--runs` times with seeds 0, 1, 2, ...
Results are written to results/raw/:

    <experiment>.csv            one row per run (accuracies, F1, ...)
    <experiment>_settings.json  the full configuration behind each label
    <experiment>_curves.npz     best fitness and swarm diversity per iteration

Run analyse_results.py afterwards to turn these into tables and figures.
"""

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments import final as final_experiments
from experiments.common import run_once
from experiments.suites import EXPERIMENTS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw")
CURVE_KEYS = ("fitness_history", "diversity_history")


def _job(args: Tuple[int, Dict, int]) -> Tuple[int, Dict]:
    """Worker: one PSO run. Defined at module level so it can be pickled."""
    index, config, seed = args
    return index, run_once(config, seed)


def run_settings(settings: List[Dict], runs: int, jobs: int) -> List[Dict]:
    """Run every setting `runs` times and return one record per run."""
    tasks = [(i, s["config"], seed) for i, s in enumerate(settings) for seed in range(runs)]
    if jobs > 1:
        with Pool(jobs) as pool:
            results = pool.map(_job, tasks, chunksize=max(1, len(tasks) // (jobs * 16)))
    else:
        results = [_job(task) for task in tasks]

    records = []
    for index, result in results:
        setting = settings[index]
        records.append({"label": setting["label"], **setting["factors"], **result})
    return records


def save_records(name: str, records: List[Dict], settings: Optional[List[Dict]] = None) -> None:
    """Write the per-run table, the settings and the per-iteration curves."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if settings is not None:
        with open(os.path.join(RESULTS_DIR, f"{name}_settings.json"), "w") as handle:
            json.dump({s["label"]: s["config"] for s in settings}, handle, indent=1)

    columns: List[str] = []
    for record in records:  # keep first-seen order, allow differing keys
        for key in record:
            if key not in CURVE_KEYS and key not in columns:
                columns.append(key)
    with open(os.path.join(RESULTS_DIR, f"{name}.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Curves: for each setting a (runs x iterations) array.
    curves: Dict[str, np.ndarray] = {}
    for label in dict.fromkeys(r["label"] for r in records):
        rows = sorted((r for r in records if r["label"] == label), key=lambda r: r["seed"])
        for key in CURVE_KEYS:
            if all(key in r for r in rows):
                curves[f"{label}::{key}"] = np.array([r[key] for r in rows], dtype=float)
    if curves:
        np.savez_compressed(os.path.join(RESULTS_DIR, f"{name}_curves.npz"), **curves)


def main() -> None:
    all_names = list(EXPERIMENTS) + list(final_experiments.EXPERIMENTS)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="all", help="name of an experiment, or 'all'")
    parser.add_argument("--runs", type=int, default=30, help="independent runs per setting (default 30)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel processes")
    parser.add_argument("--list", action="store_true", help="list the experiments and exit")
    args = parser.parse_args()

    if args.list:
        for name in EXPERIMENTS:
            print(f"{name:20s} {len(EXPERIMENTS[name]()):3d} settings   {EXPERIMENTS[name].__doc__.strip().splitlines()[0]}")
        for name, function in final_experiments.EXPERIMENTS.items():
            print(f"{name:20s}                {function.__doc__.strip().splitlines()[0]}")
        return

    names = all_names if args.experiment == "all" else [args.experiment]
    for name in names:
        started = time.time()
        settings = None
        if name in EXPERIMENTS:
            settings = EXPERIMENTS[name]()
            print(f"[{name}] {len(settings)} settings x {args.runs} runs ...", flush=True)
            records = run_settings(settings, args.runs, args.jobs)
        elif name in final_experiments.EXPERIMENTS:
            print(f"[{name}] ...", flush=True)
            records = final_experiments.EXPERIMENTS[name](args.runs, args.jobs)
        else:
            raise SystemExit(f"Unknown experiment '{name}'. Choose from: {', '.join(all_names)}")
        save_records(name, records, settings)
        print(f"[{name}] done in {time.time() - started:.0f} s -> results/raw/{name}.csv", flush=True)


if __name__ == "__main__":
    main()
