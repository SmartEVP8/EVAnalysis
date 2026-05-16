from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.run_pipeline import PipelineRunner
from analysis.scoring.simulation_scorer import compute_simulation_score

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HEADLESS_PROJECT = PROJECT_ROOT.parent / "SmartEV" / "Headless" / "Headless.csproj"

ENV_VAR_BY_WEIGHT = {
    "price_sensitivity": "COST_WEIGHT_PRICE_SENSITIVITY",
    "path_deviation": "COST_WEIGHT_PATH_DEVIATION",
    "expected_wait_time": "COST_WEIGHT_EXPECTED_WAIT_TIME",
}

# Fixed weights
FIXED_WEIGHTS: dict[str, float] = {
    "price_sensitivity": 5.0,
    "path_deviation": 50.0,
    "expected_wait_time": 50.0,
}

RESULT_FIELDNAMES = [
    "iteration",
    "run_id",
    "seed",
    "status",
    "error",
    "simulation_seconds",
    "analysis_seconds",
    "score_seconds",
    "missed_deadline_aggregate",
    "ev_wait_time_aggregate",
    "utilization_aggregate",
    "expected_wait_time_aggregate",
    "ev_aggregate",
    "station_aggregate",
    "overall_score",
    "price_sensitivity",
    "path_deviation",
    "expected_wait_time",
]

METRIC_FILENAMES = [
    "StationSnapshotMetric.parquet",
    "ChargerSnapshotMetric.parquet",
    "ArrivalAtDestinationMetric.parquet",
    "WaitTimeInQueueMetric.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous seed sweep over SmartEV with fixed cost weights."
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=43,
        help="First seed value. Each iteration increments by 1 (default: 43).",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help="Iteration number to start/resume from (default: 1).",
    )
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
        help="Optional CSV path. Defaults to <session_dir>/seeded_runs_results.csv.",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Path to an existing session directory to resume appending to.",
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


def create_session_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = PROJECT_ROOT / "runs" / "search_sessions" / f"seeded_{timestamp}"
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
    session_env: dict[str, str],
) -> Path:
    existing_run_names = list_run_dirs(perkuet_root)
    started_at = time.time()

    env = os.environ.copy()
    env.update(session_env)
    env.update({ENV_VAR_BY_WEIGHT[name]: f"{value:.8f}" for name, value in weights.items()})

    result = subprocess.run(
        ["dotnet", "run", "--project", str(headless_project), "-c", build_config],
        check=False,
        cwd=str(PROJECT_ROOT.parent),
        env=env,
    )

    if result.returncode not in [0, 139]:
        raise RuntimeError(f"dotnet run failed with exit status {result.returncode}")

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


def run_scoring(run_id: str, output_root: Path) -> dict[str, float]:
    sim_score = compute_simulation_score(
        run_id=run_id,
        source_path=str(output_root),
        output_root=output_root,
    )
    return {
        "missed_deadline_aggregate": sim_score.missed_deadline_aggregate,
        "ev_wait_time_aggregate": sim_score.ev_wait_time_aggregate,
        "utilization_aggregate": sim_score.utilization_aggregate,
        "expected_wait_time_aggregate": sim_score.expected_wait_aggregate,
        "ev_aggregate": sim_score.ev_weighted_aggregate,
        "station_aggregate": sim_score.station_weighted_aggregate,
        "overall_score": sim_score.overall_aggregate,
    }


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
    seed: int,
    run_id: str = "",
    status: str = "ok",
    error: str = "",
    simulation_seconds: float = 0.0,
    analysis_seconds: float = 0.0,
    score_seconds: float = 0.0,
    missed_deadline_aggregate: float = 0.0,
    ev_wait_time_aggregate: float = 0.0,
    utilization_aggregate: float = 0.0,
    expected_wait_time_aggregate: float = 0.0,
    ev_aggregate: float = 0.0,
    station_aggregate: float = 0.0,
    overall_score: float = 0.0,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "run_id": run_id,
        "seed": seed,
        "status": status,
        "error": error,
        "simulation_seconds": f"{simulation_seconds:.4f}",
        "analysis_seconds": f"{analysis_seconds:.4f}",
        "score_seconds": f"{score_seconds:.4f}",
        "missed_deadline_aggregate": f"{missed_deadline_aggregate:.6f}",
        "ev_wait_time_aggregate": f"{ev_wait_time_aggregate:.6f}",
        "utilization_aggregate": f"{utilization_aggregate:.6f}",
        "expected_wait_time_aggregate": f"{expected_wait_time_aggregate:.6f}",
        "ev_aggregate": f"{ev_aggregate:.6f}",
        "station_aggregate": f"{station_aggregate:.6f}",
        "overall_score": f"{overall_score:.6f}",
        "price_sensitivity": f"{weights['price_sensitivity']:.8f}",
        "path_deviation": f"{weights['path_deviation']:.8f}",
        "expected_wait_time": f"{weights['expected_wait_time']:.8f}",
    }


def run_trial(
    *,
    iteration: int,
    seed: int,
    weights: dict[str, float],
    headless_project: Path,
    build_config: str,
    perkuet_root: Path,
    output_root: Path,
    session_env: dict[str, str],
) -> dict[str, Any]:
    # Override the seed for this specific trial
    trial_env = {**session_env, "ENGINE_SEED": str(seed)}

    sim_start = time.perf_counter()
    run_dir = run_headless_once(
        headless_project=headless_project,
        build_config=build_config,
        weights=weights,
        perkuet_root=perkuet_root,
        session_env=trial_env,
    )
    validate_metrics_parquet(run_dir)
    simulation_seconds = time.perf_counter() - sim_start
    print(f"  Run: {run_dir.name} ({simulation_seconds:.2f}s)")

    analysis_start = time.perf_counter()
    run_analysis(run_dir, output_root)
    analysis_seconds = time.perf_counter() - analysis_start
    print(f"  Analysis complete ({analysis_seconds:.2f}s)")

    score_start = time.perf_counter()
    score_values = run_scoring(run_dir.name, output_root)
    score_seconds = time.perf_counter() - score_start
    print(
        f"  Scoring complete ({score_seconds:.2f}s) - "
        f"Missed deadline: {score_values['missed_deadline_aggregate']:.6f}, "
        f"EV wait: {score_values['ev_wait_time_aggregate']:.6f}, "
        f"Utilization: {score_values['utilization_aggregate']:.6f}, "
        f"Expected wait: {score_values['expected_wait_time_aggregate']:.6f}, "
        f"EV: {score_values['ev_aggregate']:.6f}, "
        f"Station: {score_values['station_aggregate']:.6f}, "
        f"Overall: {score_values['overall_score']:.6f}"
    )

    return build_result_row(
        iteration=iteration,
        weights=weights,
        seed=seed,
        run_id=run_dir.name,
        simulation_seconds=simulation_seconds,
        analysis_seconds=analysis_seconds,
        score_seconds=score_seconds,
        missed_deadline_aggregate=score_values["missed_deadline_aggregate"],
        ev_wait_time_aggregate=score_values["ev_wait_time_aggregate"],
        utilization_aggregate=score_values["utilization_aggregate"],
        expected_wait_time_aggregate=score_values["expected_wait_time_aggregate"],
        ev_aggregate=score_values["ev_aggregate"],
        station_aggregate=score_values["station_aggregate"],
        overall_score=score_values["overall_score"],
    )


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    headless_project = resolve_path(args.headless_project, must_be="file")

    if args.session_dir:
        session_dir = resolve_path(args.session_dir, must_be="dir")
        print(f"Resuming existing session in: {session_dir}")
    else:
        session_dir = create_session_dir()
        print(f"Created new session dir: {session_dir}")

    results_path = (
        args.results_file
        if args.results_file is not None
        else session_dir / "seeded_runs_results.csv"
    )
    if not results_path.is_absolute():
        results_path = (PROJECT_ROOT / results_path).resolve()

    output_root = session_dir
    perkuet_root = load_perkuet_root()

    # Base env — ENGINE_SEED is overridden per trial inside run_trial()
    session_env = {
        "SIMULATION_START_TIME_MS": "111600000",   # Monday 07:00
        "SIMULATION_END_TIME_MS": "151200000",     # Monday 18:00
        "DISABLE_FILE_LOGGING": "true",
    }

    weight_summary = ", ".join(f"{k}={v:.4f}" for k, v in FIXED_WEIGHTS.items())
    print(f"Fixed weights    : {{{weight_summary}}}")
    print(f"Start seed       : {args.start_seed}")
    print(f"Start iteration  : {args.start_iteration}")
    print(f"Headless project : {headless_project}")
    print(f"Perkuet root     : {perkuet_root}")
    print(f"Session dir      : {session_dir}")
    print(f"Results CSV      : {results_path}")
    print("Running continuously — press Ctrl+C to stop.\n")

    # Infinite seed iterator starting from the requested seed + offset
    seed_iter = itertools.count(args.start_seed + (args.start_iteration - 1))

    try:
        for iteration in itertools.count(args.start_iteration):
            seed = next(seed_iter)
            print(f"\n[iteration={iteration}] seed={seed}")

            try:
                row = run_trial(
                    iteration=iteration,
                    seed=seed,
                    weights=FIXED_WEIGHTS,
                    headless_project=headless_project,
                    build_config=args.build_config,
                    perkuet_root=perkuet_root,
                    output_root=output_root,
                    session_env=session_env,
                )
            except Exception as exc:
                print(f"  Iteration failed: {exc}")
                row = build_result_row(
                    iteration=iteration,
                    weights=FIXED_WEIGHTS,
                    seed=seed,
                    status="error",
                    error=str(exc),
                )

            append_result_row(results_path, row)

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results are already saved.")


if __name__ == "__main__":
    main()