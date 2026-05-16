"""
Re-scores all simulation runs in a grid search session directory
using an optionally updated ScoringConfig.

Usage:
    python rescoring.py <session_dir>
    python rescoring.py <session_dir> --config my_scoring.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from analysis.scoring.default_scores import DEFAULT_SCORING_CONFIG, ScoringConfig
from grid_search import run_scoring

_SENTINEL_FILE = "simulation_score.json" 

def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / _SENTINEL_FILE).exists()


def load_scoring_config(config_path: Path) -> ScoringConfig:
    """
    Loads a ScoringConfig from a TOML file. Only the keys present are overridden;
    everything else falls back to DEFAULT_SCORING_CONFIG.

    Example TOML:
        [ev_metric_weights]
        path_deviation = 2
        delta_arrival = 3
        ev_wait_time = 4
        missed_deadline = 1

        [station_metric_weights]
        utilization = 1
        expected_wait_time = 2

        [group_weights]
        ev = 1
        station = 2
    """
    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)

    def _parse_buckets(raw_list: list[dict]) -> list[tuple[float, int]]:
        result = []
        for entry in raw_list:
            upper = float("inf") if entry["upper"] == "inf" else float(entry["upper"])
            result.append((upper, int(entry["weight"])))
        return result
    
    def _parse_percentile_buckets(raw_list: list[dict]) -> list[tuple[str, int]]:
        return [(entry["percentile"], int(entry["weight"])) for entry in raw_list]

    d = DEFAULT_SCORING_CONFIG

    return ScoringConfig(
        path_deviation_buckets=(
            _parse_buckets(raw["path_deviation_buckets"])
            if "path_deviation_buckets" in raw
            else d.path_deviation_buckets
        ),
        delta_arrival_buckets=(
            _parse_buckets(raw["delta_arrival_buckets"])
            if "delta_arrival_buckets" in raw
            else d.delta_arrival_buckets
        ),
        wait_time_buckets=(
            _parse_percentile_buckets(raw["wait_time_buckets"])
            if "wait_time_buckets" in raw
            else d.wait_time_buckets
        ),
        expected_wait_time_buckets=(
            _parse_percentile_buckets(raw["expected_wait_time_buckets"])
            if "expected_wait_time_buckets" in raw
            else d.expected_wait_time_buckets
        ),
        ev_metric_weights=raw.get("ev_metric_weights", d.ev_metric_weights),
        station_metric_weights=raw.get("station_metric_weights", d.station_metric_weights),
        group_weights=raw.get("group_weights", d.group_weights),
    )


def rescorer(session_path: Path, scoring_config: ScoringConfig = DEFAULT_SCORING_CONFIG) -> None:
    if not session_path.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_path}")

    run_dirs = sorted(d for d in session_path.iterdir() if _is_run_dir(d))

    if not run_dirs:
        print(f"No run directories found under {session_path}")
        return

    print(f"Found {len(run_dirs)} run(s) to rescore in {session_path}\n")

    for run_dir in run_dirs:
        print(f"  Rescoring {run_dir.name} ...", end=" ", flush=True)
        try:
            scores = run_scoring(run_dir.name, session_path, scoring_config=scoring_config)
            print(f"overall={scores['overall_score']:.6f}")
        except Exception as exc:
            print(f"FAILED: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore all runs in a grid search session.")
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to the session directory (absolute or relative to cwd).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="TOML_FILE",
        help="Optional scoring config TOML. Overrides defaults for any keys present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve relative to cwd, not the script location
    session_path = args.session_dir.resolve()

    if args.config is not None:
        config_path = args.config.resolve()
        scoring_config = load_scoring_config(config_path)
        print(f"Using scoring config: {config_path}")
    else:
        scoring_config = DEFAULT_SCORING_CONFIG
        print("Using default scoring config")

    rescorer(session_path, scoring_config=scoring_config)


if __name__ == "__main__":
    main()