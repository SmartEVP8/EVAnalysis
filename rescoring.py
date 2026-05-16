"""
Re-scores all simulation runs in a grid search session directory
using an optionally updated ScoringConfig.

Usage:
    python rescoring.py <session_dir>
    python rescoring.py <session_dir> --config my_scoring.toml
"""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path

from analysis.scoring.default_scores import DEFAULT_SCORING_CONFIG, ScoringConfig
from grid_search import RESULT_FIELDNAMES, run_scoring

_SENTINEL_FILE = "simulation_score.json"

SCORE_COLUMNS = [
    "missed_deadline_aggregate",
    "ev_wait_time_aggregate",
    "utilization_aggregate",
    "expected_wait_time_aggregate",
    "ev_aggregate",
    "station_aggregate",
    "overall_score",
]


def _parse_buckets(raw_list: list[dict]) -> list[tuple[float, int]]:
    result = []
    for entry in raw_list:
        upper = float("inf") if entry["upper"] == "inf" else float(entry["upper"])
        result.append((upper, int(entry["weight"])))
    return result


def _parse_percentile_buckets(raw_list: list[dict]) -> list[tuple[str, int]]:
    return [(entry["percentile"], int(entry["weight"])) for entry in raw_list]


def load_scoring_config(config_path: Path) -> ScoringConfig:
    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)

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


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / _SENTINEL_FILE).exists()


def _read_score_json(run_dir: Path) -> dict:
    """Read the freshly written simulation_score.json for this run."""
    with open(run_dir / "simulation_score.json", encoding="utf-8") as fh:
        return json.load(fh)


def _score_columns_from_json(data: dict) -> dict[str, str]:
    """Extract the score columns that appear in the CSV from a simulation_score.json."""
    ev = data["ev_scores"]
    station = data["station_scores"]
    return {
        "missed_deadline_aggregate": f"{ev['per_metric']['missed_deadline']['aggregate_score']:.6f}",
        "ev_wait_time_aggregate":    f"{ev['per_metric']['ev_wait_time']['aggregate_score']:.6f}",
        "utilization_aggregate":     f"{station['per_metric']['utilization']['aggregate_score']:.6f}",
        "expected_wait_time_aggregate": f"{station['per_metric']['expected_wait_time']['aggregate_score']:.6f}",
        "ev_aggregate":              f"{ev['weighted_aggregate']:.6f}",
        "station_aggregate":         f"{station['weighted_aggregate']:.6f}",
        "overall_score":             f"{data['overall_aggregate']:.6f}",
    }


def write_rescored_csv(session_path: Path, rescored: dict[str, dict[str, str]]) -> Path:
    """
    Read the original grid_search_results.csv, patch the score columns with
    fresh values, and write grid_search_results_rescored.csv (always overwritten).

    rescored maps run_id -> {column: value} for every score column.
    """
    original_csv = session_path / "grid_search_results.csv"
    output_csv = session_path / "grid_search_results_rescored.csv"

    if not original_csv.exists():
        print(f"  [CSV] No grid_search_results.csv found in {session_path}, skipping.")
        return output_csv

    with open(original_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        run_id = row["run_id"]
        if run_id in rescored:
            row.update(rescored[run_id])

    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[CSV] Wrote rescored results to {output_csv}")
    return output_csv


def rescorer(session_path: Path, scoring_config: ScoringConfig = DEFAULT_SCORING_CONFIG) -> None:
    if not session_path.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_path}")

    run_dirs = sorted(d for d in session_path.iterdir() if _is_run_dir(d))

    if not run_dirs:
        print(f"No run directories found under {session_path}")
        return

    print(f"Found {len(run_dirs)} run(s) to rescore in {session_path}\n")

    rescored: dict[str, dict[str, str]] = {}

    for run_dir in run_dirs:
        print(f"  Rescoring {run_dir.name} ...", end=" ", flush=True)
        try:
            run_scoring(run_dir.name, session_path, scoring_config=scoring_config)
            score_data = _read_score_json(run_dir)
            rescored[run_dir.name] = _score_columns_from_json(score_data)
            print(f"overall={rescored[run_dir.name]['overall_score']}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    write_rescored_csv(session_path, rescored)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore all runs in a grid search session.")
    parser.add_argument(
        "session_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the session directory. Defaults to the most recently created session.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="TOML_FILE",
        help="Optional scoring config TOML. Overrides defaults for any keys present.",
    )
    return parser.parse_args()


def latest_session_dir() -> Path:
    sessions_root = Path("runs/search_sessions")
    candidates = [d for d in sessions_root.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No session directories found under {sessions_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main() -> None:
    args = parse_args()

    session_path = (
        args.session_dir.resolve()
        if args.session_dir
        else latest_session_dir().resolve()
    )

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