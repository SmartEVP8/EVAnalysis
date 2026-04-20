from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import subprocess
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.run_pipeline import PipelineRunner

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HEADLESS_PROJECT = PROJECT_ROOT.parent / "SmartEV" / "Headless" / "Headless.csproj"
DEFAULT_RANDOM_RESULTS_PATH = PROJECT_ROOT / "runs" / "random_search_results.csv"
DEFAULT_GRID_RESULTS_PATH   = PROJECT_ROOT / "runs" / "grid_search_results.csv"

ENV_VAR_BY_WEIGHT = {
    "price_sensitivity": "COST_WEIGHT_PRICE_SENSITIVITY",
    "path_deviation": "COST_WEIGHT_PATH_DEVIATION",
    "effective_queue_size": "COST_WEIGHT_EFFECTIVE_QUEUE_SIZE",
    "urgency": "COST_WEIGHT_URGENCY",
    "expected_wait_time": "COST_WEIGHT_EXPECTED_WAIT_TIME",
}

RESULT_FIELDNAMES = [
    "iteration",
    "run_id",
    "status",
    "error",
    "simulation_seconds",
    "analysis_seconds",
    "price_sensitivity",
    "path_deviation",
    "effective_queue_size",
    "urgency",
    "expected_wait_time",
]

METRIC_FILENAMES = [
    "StationSnapshotMetric.parquet",
    "ChargerSnapshotMetric.parquet",
    "ArrivalAtDestinationMetric.parquet",
]


@dataclass(frozen=True)
class WeightRanges:
    price_sensitivity: tuple[float, float]
    path_deviation: tuple[float, float]
    effective_queue_size: tuple[float, float]
    urgency: tuple[float, float]
    expected_wait_time: tuple[float, float]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weight search over SmartEV cost weights.")

    parser.add_argument("--iterations", type=int, default=10, help="Number of random trials to run.")
    parser.add_argument(
        "--headless-project",
        type=Path,
        default=DEFAULT_HEADLESS_PROJECT,
        help="Path to Headless.csproj",
    )
    parser.add_argument(
        "--build-config",
        type=str,
        default="Release",
        choices=["Debug", "Release"],
        help="dotnet build configuration.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Optional CSV path. If omitted, defaults to random/grid results file based on --strategy.",
    )
    parser.add_argument("--price-sensitivity-range", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--path-deviation-range", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--effective-queue-size-range", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--urgency-range", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--expected-wait-time-range", type=float, nargs=2, default=(0.0, 1.0))

    parser.add_argument(
        "--strategy",
        "--s",
        dest="strategy",
        type=str,
        default="random",
        choices=["random", "grid"],
        help="Search strategy to use (default: random).",
    )

    parser.add_argument(
        "--points-per-axis",
        type=int,
        default=5,
        help="Grid search: number of evenly-spaced values per weight axis (e.g. 5 → [0, 0.25, 0.5, 0.75, 1]). "
             "Total trials = points_per_axis ** 5.",
    )

    return parser.parse_args()


def load_perkuet_root() -> Path:
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as fh:
        config = tomllib.load(fh)

    perkuet_dir = Path(config["paths"]["perkuet_dir"])
    if not perkuet_dir.is_absolute():
        perkuet_dir = (PROJECT_ROOT / perkuet_dir).resolve()

    if not perkuet_dir.is_dir():
        raise FileNotFoundError(f"perkuet_dir does not exist: {perkuet_dir}")

    return perkuet_dir


def resolve_path(path: Path, *, must_be: str = "file") -> Path:
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    if must_be == "file" and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if must_be == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    return path


def validate_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = bounds
    if low > high:
        raise ValueError(f"Invalid range for {name}: min ({low}) > max ({high})")
    return low, high


def build_ranges(args: argparse.Namespace) -> WeightRanges:
    return WeightRanges(
        price_sensitivity=validate_range("price_sensitivity", tuple(args.price_sensitivity_range)),
        path_deviation=validate_range("path_deviation", tuple(args.path_deviation_range)),
        effective_queue_size=validate_range("effective_queue_size", tuple(args.effective_queue_size_range)),
        urgency=validate_range("urgency", tuple(args.urgency_range)),
        expected_wait_time=validate_range("expected_wait_time", tuple(args.expected_wait_time_range)),
    )


def sample_weights(rng: random.Random, ranges: WeightRanges) -> dict[str, float]:
    return {
        "price_sensitivity": rng.uniform(*ranges.price_sensitivity),
        "path_deviation": rng.uniform(*ranges.path_deviation),
        "effective_queue_size": rng.uniform(*ranges.effective_queue_size),
        "urgency": rng.uniform(*ranges.urgency),
        "expected_wait_time": rng.uniform(*ranges.expected_wait_time),
    }

def random_search_weights(args: argparse.Namespace, ranges: WeightRanges) -> list[dict[str, float]]:
    rng = random.Random()
    return [sample_weights(rng, ranges) for _ in range(args.iterations)]


def build_grid(points_per_axis: int) -> list[dict[str, float]]:
    if points_per_axis < 2:
        raise ValueError("--points-per-axis must be >= 2")

    axis_values = [i / (points_per_axis - 1) for i in range(points_per_axis)]
    keys = list(ENV_VAR_BY_WEIGHT) 
    
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(axis_values, repeat=len(keys))
    ]


def create_search_session_dir(strategy: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = PROJECT_ROOT / "runs" / "search_sessions" / f"{strategy}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def list_run_dirs(perkuet_root: Path) -> set[str]:
    return {path.name for path in perkuet_root.iterdir() if path.is_dir()}


def run_headless_once(
    *,
    headless_project: Path,
    build_config: str,
    weights: dict[str, float],
    perkuet_root: Path,
) -> Path:
    existing_run_names = list_run_dirs(perkuet_root)
    started_at = time.time()

    env = os.environ.copy()
    env.update({ENV_VAR_BY_WEIGHT[name]: f"{value:.8f}" for name, value in weights.items()})

    subprocess.run(
        ["dotnet", "run", "--project", str(headless_project), "-c", build_config],
        check=True,
        cwd=str(PROJECT_ROOT.parent),
        env=env,
    )

    all_run_dirs = [path for path in perkuet_root.iterdir() if path.is_dir()]

    new_dirs = [path for path in all_run_dirs if path.name not in existing_run_names]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)

    recent_dirs = [path for path in all_run_dirs if path.stat().st_mtime >= started_at - 1]
    if recent_dirs:
        return max(recent_dirs, key=lambda p: p.stat().st_mtime)

    raise RuntimeError("Could not determine which simulation run directory was created.")


def metric_paths(run_dir: Path) -> list[Path]:
    return [run_dir / name for name in METRIC_FILENAMES]


# Verifies that metric files exist, are non-empty, and contain valid Parquet data.
def validate_metrics_parquet(run_dir: Path) -> None:
    paths = metric_paths(run_dir)
    missing = []
    empty = []

    for path in paths:
        if not path.exists():
            missing.append(path.name)
        elif path.stat().st_size == 0:
            empty.append(path.name)

    if missing or empty:
        error_msg = f"Simulation failed to output valid metrics. run={run_dir}"
        if missing:
            error_msg += f"\n  Missing: {', '.join(missing)}"
        if empty:
            error_msg += f"\n  Empty: {', '.join(empty)}"
        raise RuntimeError(error_msg)

    for path in paths:
        try:
            pl.read_parquet(path, n_rows=1)
        except Exception as e:
            raise RuntimeError(f"Corrupt Parquet file '{path.name}': {e}")

def run_analysis(run_dir: Path, output_root: Path) -> None:
    PipelineRunner(run_dir, output_root=output_root).run_all()

def append_result_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def build_result_row(
    *,
    iteration: int,
    weights: dict[str, float],
    run_id: str = "",
    status: str = "ok",
    error: str = "",
    simulation_seconds: float = 0.0,
    analysis_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "run_id": run_id,
        "status": status,
        "error": error,
        "simulation_seconds": f"{simulation_seconds:.4f}",
        "analysis_seconds": f"{analysis_seconds:.4f}",
        "price_sensitivity": f"{weights['price_sensitivity']:.8f}",
        "path_deviation": f"{weights['path_deviation']:.8f}",
        "effective_queue_size": f"{weights['effective_queue_size']:.8f}",
        "urgency": f"{weights['urgency']:.8f}",
        "expected_wait_time": f"{weights['expected_wait_time']:.8f}",
    }

def run_trial(
    *,
    iteration: int,
    weights: dict[str, float],
    headless_project: Path,
    build_config: str,
    perkuet_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    sim_start = time.perf_counter()
    
    run_dir = run_headless_once(
        headless_project=headless_project,
        build_config=build_config,
        weights=weights,
        perkuet_root=perkuet_root,
    )
    validate_metrics_parquet(run_dir)

    simulation_seconds = time.perf_counter() - sim_start
    print(f"  Run: {run_dir.name} ({simulation_seconds:.2f}s)")

    analysis_start = time.perf_counter()
    run_analysis(run_dir, output_root)
    analysis_seconds = time.perf_counter() - analysis_start
    print(f"  Analysis complete ({analysis_seconds:.2f}s)")

    return build_result_row(
        iteration=iteration,
        weights=weights,
        run_id=run_dir.name,
        simulation_seconds=simulation_seconds,
        analysis_seconds=analysis_seconds,
    )

def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")

    headless_project = resolve_path(args.headless_project, must_be="file")
    session_dir = create_search_session_dir(args.strategy)

    if args.results_file is None:
        results_path = session_dir / f"{args.strategy}_search_results.csv"
    else:
        results_path = args.results_file if args.results_file.is_absolute() else (PROJECT_ROOT / args.results_file).resolve()
    output_root = session_dir
    perkuet_root = load_perkuet_root()
    ranges = build_ranges(args)

    if args.strategy == "random":
        all_weights = random_search_weights(args, ranges)
    else:
        all_weights = build_grid(args.points_per_axis)

    total = len(all_weights)
    print(f"Running {total} trials")
    print(f"Headless project : {headless_project}")
    print(f"Perkuet root     : {perkuet_root}")
    print(f"Session dir      : {session_dir}")
    print(f"Results CSV      : {results_path}")

    try:
        for iteration, weights in enumerate(all_weights, start=1):
            weight_summary = ", ".join(f"{k}={v:.4f}" for k, v in weights.items())
            print(f"\n[{iteration}/{total}] weights={{{weight_summary}}}")

            try:
                row = run_trial(
                    iteration=iteration,
                    weights=weights,
                    headless_project=headless_project,
                    build_config=args.build_config,
                    perkuet_root=perkuet_root,
                    output_root=output_root,
                )
            except Exception as exc:
                print(f"  Iteration failed: {exc}")
                row = build_result_row(iteration=iteration, weights=weights, status="error", error=str(exc))

            append_result_row(results_path, row)

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results are already saved.")
        return

    print("\nTrials complete.")


if __name__ == "__main__":
    main()