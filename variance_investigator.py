"""
variance_investigator.py

Finds the most recently modified session directory under runs/search_sessions/,
scores every run found there against a set of scoring configurations, then prints
a variance report to see whether the scoring system meaningfully
differentiates between simulation runs.

How to run
-----
    uv run variance_investigator.py

Each "scoring configuration" is a dict that can override any combination of:
    - EV_METRIC_WEIGHTS        (dict[str, int])
    - STATION_METRIC_WEIGHTS   (dict[str, int])
    - GROUP_WEIGHTS            (dict[str, int])

Add or remove entries in SCORING_CONFIGS to explore the space.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).parent
RUNS_ROOT = ROOT / "runs" / "search_sessions"

SCORE_METRICS = [
    "path_deviation_score",
    "delta_arrival_score",
    "ev_wait_time_score",
    "missed_deadline_score",
    "utilization_score",
    "expected_wait_score",
    "ev_weighted_score",
    "station_weighted_score",
    "combined_score",
]

# ── import the modules we're going to monkeypatch ───────────────────────────
import analysis.scoring.ev_scorer as ev_mod
import analysis.scoring.station_scorer as station_mod
import analysis.scoring.simulation_scorer as simulation_mod

from concurrent.futures import ProcessPoolExecutor, as_completed

from helpers.variance_configs import SCORING_CONFIGS

# ────────────────────────────────────────────────────────────────────────────
# Session discovery
# ────────────────────────────────────────────────────────────────────────────

def find_latest_session() -> Path:
    sessions = sorted(
        (path for path in RUNS_ROOT.iterdir() if path.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        raise FileNotFoundError(f"No session directories found under {RUNS_ROOT}")
    return sessions[0]


# ────────────────────────────────────────────────────────────────────────────
# Monkeypatching helpers
# ────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def apply_config(config: dict[str, Any]):
    """
    Temporarily overrides module-level weight constants in the three scorer
    modules, runs the body of the `with` block, then restores the originals.

    Decay constants are intentionally never touched here.
    """
    # save originals
    original_weights = {
        "ev_EV_METRIC_WEIGHTS": copy.deepcopy(ev_mod.EV_METRIC_WEIGHTS),
        "st_STATION_METRIC_WEIGHTS": copy.deepcopy(station_mod.STATION_METRIC_WEIGHTS),
        "sim_GROUP_WEIGHTS": copy.deepcopy(simulation_mod.GROUP_WEIGHTS),
        "sim_TOTAL_GROUP_WEIGHT": simulation_mod.TOTAL_GROUP_WEIGHT,
    }

    try:
        if "EV_METRIC_WEIGHTS" in config:
            ev_mod.EV_METRIC_WEIGHTS.clear()
            ev_mod.EV_METRIC_WEIGHTS.update(config["EV_METRIC_WEIGHTS"])

        if "STATION_METRIC_WEIGHTS" in config:
            station_mod.STATION_METRIC_WEIGHTS.clear()
            station_mod.STATION_METRIC_WEIGHTS.update(config["STATION_METRIC_WEIGHTS"])

        if "GROUP_WEIGHTS" in config:
            simulation_mod.GROUP_WEIGHTS.clear()
            simulation_mod.GROUP_WEIGHTS.update(config["GROUP_WEIGHTS"])
            simulation_mod.TOTAL_GROUP_WEIGHT = sum(simulation_mod.GROUP_WEIGHTS.values())

        yield

    finally:
        ev_mod.EV_METRIC_WEIGHTS.clear()
        ev_mod.EV_METRIC_WEIGHTS.update(original_weights["ev_EV_METRIC_WEIGHTS"])

        station_mod.STATION_METRIC_WEIGHTS.clear()
        station_mod.STATION_METRIC_WEIGHTS.update(original_weights["st_STATION_METRIC_WEIGHTS"])

        simulation_mod.GROUP_WEIGHTS.clear()
        simulation_mod.GROUP_WEIGHTS.update(original_weights["sim_GROUP_WEIGHTS"])
        simulation_mod.TOTAL_GROUP_WEIGHT = original_weights["sim_TOTAL_GROUP_WEIGHT"]


# ────────────────────────────────────────────────────────────────────────────
# Output paths
# ────────────────────────────────────────────────────────────────────────────

def variance_investigation_dir(session_dir: Path, run_id: str, config_name: str) -> Path:
    return session_dir / "variance_investigations" / run_id / config_name


# ────────────────────────────────────────────────────────────────────────────
# Data collection
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    run_id: str
    config_name: str
    score: simulation_mod.SimulationScore | None = None
    error: str | None = None

def score_and_write(run_id: str, config: dict, session_dir: Path) -> RunResult:
    """Top-level function so it's picklable for multiprocessing."""
    result = score_run(run_id, config, output_root=session_dir)
    write_run_outputs(result, session_dir)
    return result


def score_run(run_id: str, config: dict[str, Any], output_root: Path) -> RunResult:
    """
    Scores a single run under a single config.

    Returns a RunResult holding the full SimulationScore object so callers
    can write whatever outputs they need from it.
    """
    try:
        with apply_config(config):
            ev_scores      = simulation_mod.compute_ev_scores(run_id, output_root)
            station_scores = simulation_mod.compute_station_scores(run_id, output_root)
            score = simulation_mod.SimulationScore(
                run_id=run_id,
                source_path=str(output_root / run_id),
                ev_scores=ev_scores,
                station_scores=station_scores,
            )
        return RunResult(run_id=run_id, config_name=config["name"], score=score)
    except Exception as exc:
        print(f"  [!] {run_id} / {config['name']}: {exc}")
        return RunResult(run_id=run_id, config_name=config["name"], error=str(exc))


# ────────────────────────────────────────────────────────────────────────────
# Per-run output writing
# ────────────────────────────────────────────────────────────────────────────

def write_run_outputs(result: RunResult, session_dir: Path) -> None:
    """
    Writes simulation_score.json and simulation_score.parquet for one
    (run_id, config) pair under {session_dir}/variance_investigations/{run_id}/{config_name}/.
    """
    if result.score is None:
        return

    out_dir = variance_investigation_dir(session_dir, result.run_id, result.config_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.score.write_json(out_dir / "simulation_score.json")
    result.score.write_parquet(out_dir / "simulation_score.parquet")


# ────────────────────────────────────────────────────────────────────────────
# Comparison parquet
# ────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "overall_aggregate",
    "ev_weighted_aggregate",
    "station_weighted_aggregate",
    "path_deviation_aggregate",
    "delta_arrival_aggregate",
    "ev_wait_time_aggregate",
    "missed_deadline_aggregate",
    "utilization_aggregate",
    "expected_wait_aggregate",
]


def build_comparison_df(results: list[RunResult], configs: list[dict[str, Any]]) -> pl.DataFrame:
    """
    One row per (run_id, config) with:
        - config name + raw weight values
        - for each metric: min, max, spread (max - min) across snapshots
    """
    config_by_name = {c["name"]: c for c in configs}
    rows = []

    for result in results:
        config = config_by_name[result.config_name]
        ev_weights  = config.get("EV_METRIC_WEIGHTS", {})
        station_weights  = config.get("STATION_METRIC_WEIGHTS", {})
        group_weights = config.get("GROUP_WEIGHTS", {})

        row: dict = {
            "run_id":  result.run_id,
            "config":  result.config_name,
            "error":   result.error or "",
            "config_ev_weight_path_deviation":  ev_weights.get("path_deviation"),
            "config_ev_weight_delta_arrival":   ev_weights.get("delta_arrival"),
            "config_ev_weight_ev_wait_time":    ev_weights.get("ev_wait_time"),
            "config_ev_weight_missed_deadline": ev_weights.get("missed_deadline"),
            "config_st_weight_utilization":     station_weights.get("utilization"),
            "config_st_weight_expected_wait":   station_weights.get("expected_wait_time"),
            "config_group_weight_ev":           group_weights.get("ev"),
            "config_group_weight_station":      group_weights.get("station"),
        }

        if result.score is None:
            for metric in SCORE_METRICS:
                row[f"{metric}_min"]    = None
                row[f"{metric}_max"]    = None
                row[f"{metric}_spread"] = None
        else:
            per_snapshot = result.score.per_snapshot
            for metric in SCORE_METRICS:
                column = per_snapshot[metric]
                lowest = column.min()
                highest = column.max()
                row[f"{metric}_min"]    = lowest
                row[f"{metric}_max"]    = highest
                row[f"{metric}_spread"] = (highest - lowest) if (highest is not None and lowest is not None) else None

        rows.append(row)

    return pl.DataFrame(rows)


def write_comparison_parquet(df: pl.DataFrame, session_dir: Path) -> Path:
    out_path = session_dir / "variance_investigations" / "comparison.parquet"
    (session_dir / "variance_investigations").mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"[Investigator] Wrote {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Variance parquet
# ────────────────────────────────────────────────────────────────────────────

def build_variance_df(
    results: list[RunResult],
    configs: list[dict[str, Any]],
) -> pl.DataFrame:
    """
    One row per (run_id, config, metric) with:
        - simtime_ms and score of the snapshot with the highest score for that metric
        - simtime_ms and score of the snapshot with the lowest score for that metric
        - spread (max_score - min_score)
        - baseline_spread for that same (run_id, metric)
        - spread_vs_baseline = spread - baseline_spread
    """
    config_by_name = {c["name"]: c for c in configs}
    result_index: dict[tuple[str, str], RunResult] = {
        (r.run_id, r.config_name): r for r in results
    }

    rows = []

    for result in results:
        if result.score is None:
            continue

        per_snapshot = result.score.per_snapshot
        run_id = result.run_id
        config = result.config_name

        baseline_result = result_index.get((run_id, "baseline"))
        baseline_per_snapshot = baseline_result.score.per_snapshot if (baseline_result and baseline_result.score) else None

        for metric in SCORE_METRICS:
            column = per_snapshot[metric]
            highest  = column.max()
            lowest  = column.min()

            if highest is None or lowest is None:
                continue

            max_simtime_ms = per_snapshot.filter(pl.col(metric) == highest)["simtime_ms"][0]
            min_simtime_ms = per_snapshot.filter(pl.col(metric) == lowest)["simtime_ms"][0]
            spread = highest - lowest

            baseline_spread: float | None = None
            if baseline_per_snapshot is not None:
                baseline_highest = baseline_per_snapshot[metric].max()
                baseline_lowest = baseline_per_snapshot[metric].min()
                if baseline_highest is not None and baseline_lowest is not None:
                    baseline_spread = baseline_highest - baseline_lowest

            rows.append({
                "run_id":           run_id,
                "config":           config,
                "metric":           metric,
                "max_score":        highest,
                "max_simtime_ms":   max_simtime_ms,
                "min_score":        lowest,
                "min_simtime_ms":   min_simtime_ms,
                "spread":           spread,
                "baseline_spread":  baseline_spread,
                "spread_vs_baseline": (spread - baseline_spread) if baseline_spread is not None else None,
            })

    return pl.DataFrame(rows)


def write_variance_parquet(df: pl.DataFrame, session_dir: Path) -> Path:
    out_path = session_dir / "variance_investigations" / "variance.parquet"
    (session_dir / "variance_investigations").mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"[Investigator] Wrote {out_path}")
    return out_path

# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    session_dir = find_latest_session()
    print(f"Session: {session_dir}")

    run_dirs = sorted(
        path for path in session_dir.iterdir()
        if path.is_dir() and path.name != "variance_investigations"
    )

    if not run_dirs:
        print(f"No run directories found under {session_dir}. Exiting.")
        return

    run_ids = [path.name for path in run_dirs]
    print(f"Found {len(run_ids)} run(s): {', '.join(run_ids)}")
    print(f"Scoring against {len(SCORING_CONFIGS)} config(s)...\n")

    tasks = [(run_id, config) for run_id in run_ids for config in SCORING_CONFIGS]
    results: list[RunResult] = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(score_and_write, run_id, config, session_dir): (run_id, config["name"])
            for run_id, config in tasks
        }
        for future in as_completed(futures):
            run_id, config_name = futures[future]
            print(f"  Done: {run_id} / '{config_name}'")
            results.append(future.result())

    comparison_df = build_comparison_df(results, SCORING_CONFIGS)
    write_comparison_parquet(comparison_df, session_dir)

    scores_with_data = [r for r in results if r.score is not None]
    print(f"Results with score data: {len(scores_with_data)} / {len(results)}")
    print(f"Example per_snapshot shape: {scores_with_data[0].score.per_snapshot.shape if scores_with_data else 'N/A'}")

    variance_df = build_variance_df(results, SCORING_CONFIGS)
    write_variance_parquet(variance_df, session_dir)

if __name__ == "__main__":
    main()