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

WARMUP_MS: int = (3 * 60 * 60 * 1000) - 1200000  # 3 hours minus 20 minutes to exclude the initial ramp-up and ramp-down periods

# ── import the modules we're going to monkeypatch ───────────────────────────
import analysis.scoring.ev_scorer as ev_mod
import analysis.scoring.station_scorer as st_mod
import analysis.scoring.simulation_scorer as sim_mod

from concurrent.futures import ProcessPoolExecutor, as_completed

from helpers.variance_configs import SCORING_CONFIGS

# ────────────────────────────────────────────────────────────────────────────
# Session discovery
# ────────────────────────────────────────────────────────────────────────────

def find_latest_session() -> Path:
    sessions = sorted(
        (p for p in RUNS_ROOT.iterdir() if p.is_dir()),
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
    orig = {
        "ev_EV_METRIC_WEIGHTS":      copy.deepcopy(ev_mod.EV_METRIC_WEIGHTS),
        "st_STATION_METRIC_WEIGHTS": copy.deepcopy(st_mod.STATION_METRIC_WEIGHTS),
        "sim_GROUP_WEIGHTS":         copy.deepcopy(sim_mod.GROUP_WEIGHTS),
        "sim_TOTAL_GROUP_WEIGHT":    sim_mod.TOTAL_GROUP_WEIGHT,
    }

    try:
        if "EV_METRIC_WEIGHTS" in config:
            ev_mod.EV_METRIC_WEIGHTS.clear()
            ev_mod.EV_METRIC_WEIGHTS.update(config["EV_METRIC_WEIGHTS"])

        if "STATION_METRIC_WEIGHTS" in config:
            st_mod.STATION_METRIC_WEIGHTS.clear()
            st_mod.STATION_METRIC_WEIGHTS.update(config["STATION_METRIC_WEIGHTS"])

        if "GROUP_WEIGHTS" in config:
            sim_mod.GROUP_WEIGHTS.clear()
            sim_mod.GROUP_WEIGHTS.update(config["GROUP_WEIGHTS"])
            sim_mod.TOTAL_GROUP_WEIGHT = sum(sim_mod.GROUP_WEIGHTS.values())

        yield

    finally:
        ev_mod.EV_METRIC_WEIGHTS.clear()
        ev_mod.EV_METRIC_WEIGHTS.update(orig["ev_EV_METRIC_WEIGHTS"])

        st_mod.STATION_METRIC_WEIGHTS.clear()
        st_mod.STATION_METRIC_WEIGHTS.update(orig["st_STATION_METRIC_WEIGHTS"])

        sim_mod.GROUP_WEIGHTS.clear()
        sim_mod.GROUP_WEIGHTS.update(orig["sim_GROUP_WEIGHTS"])
        sim_mod.TOTAL_GROUP_WEIGHT = orig["sim_TOTAL_GROUP_WEIGHT"]


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
    score: sim_mod.SimulationScore | None = None
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
            ev_scores      = sim_mod.compute_ev_scores(run_id, output_root)
            station_scores = sim_mod.compute_station_scores(run_id, output_root)
            score = sim_mod.SimulationScore(
                run_id=run_id,
                source_path=str(output_root / run_id),
                ev_scores=ev_scores,
                station_scores=station_scores,
            )
        return RunResult(run_id=run_id, config_name=config["name"], score=score)
    except Exception as exc:  # noqa: BLE001
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


def result_to_flat_row(result: RunResult, config: dict[str, Any]) -> dict:
    """
    Flattens a RunResult into a single dict row for the comparison parquet.
    Mirrors the structure of simulation_score.json so nothing is lost.
    """
    ev_weights  = config.get("EV_METRIC_WEIGHTS", {})
    st_weights  = config.get("STATION_METRIC_WEIGHTS", {})
    grp_weights = config.get("GROUP_WEIGHTS", {})

    row: dict = {
        "run_id": result.run_id,
        "config": result.config_name,
        "error":  result.error or "",
        "config_ev_weight_path_deviation":  ev_weights.get("path_deviation"),
        "config_ev_weight_delta_arrival":   ev_weights.get("delta_arrival"),
        "config_ev_weight_ev_wait_time":    ev_weights.get("ev_wait_time"),
        "config_ev_weight_missed_deadline": ev_weights.get("missed_deadline"),
        "config_st_weight_utilization":     st_weights.get("utilization"),
        "config_st_weight_expected_wait":   st_weights.get("expected_wait_time"),
        "config_group_weight_ev":           grp_weights.get("ev"),
        "config_group_weight_station":      grp_weights.get("station"),
    }

    if result.score is None:
        for col in METRIC_COLS:
            row[col] = None
        return row

    s = result.score
    row.update({
        "overall_aggregate":          s.overall_aggregate,
        "ev_weighted_aggregate":      s.ev_weighted_aggregate,
        "path_deviation_aggregate":   s.path_deviation_aggregate,
        "delta_arrival_aggregate":    s.delta_arrival_aggregate,
        "ev_wait_time_aggregate":     s.ev_wait_time_aggregate,
        "missed_deadline_aggregate":  s.missed_deadline_aggregate,
        "station_weighted_aggregate": s.station_weighted_aggregate,
        "utilization_aggregate":      s.utilization_aggregate,
        "expected_wait_aggregate":    s.expected_wait_aggregate,
    })
    return row


def build_comparison_df(results: list[RunResult], configs: list[dict[str, Any]]) -> pl.DataFrame:
    """One row per (run_id, config) with all aggregates and config params."""
    config_by_name = {c["name"]: c for c in configs}
    return pl.DataFrame([
        result_to_flat_row(r, config_by_name[r.config_name])
        for r in results
    ])


def write_comparison_parquet(df: pl.DataFrame, session_dir: Path) -> Path:
    out_path = session_dir / "variance_investigations" / "comparison.parquet"
    (session_dir / "variance_investigations").mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"[Investigator] Wrote {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Variance parquet
# ────────────────────────────────────────────────────────────────────────────

def build_variance_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Produces a tidy table with one row per (run_id, config) containing:
        run_id            – the simulation run
        config            – the scoring configuration name
        overall_aggregate – the run's overall score under that config
        delta_from_baseline – difference vs the same run's baseline score
                              (positive = scored higher than baseline)

    Rows where either the config score or the baseline score is null are
    kept but will have a null delta.
    """
    baseline = (
        df.filter(pl.col("config") == "baseline")
        .select(["run_id", pl.col("overall_aggregate").alias("baseline_aggregate")])
    )

    return (
        df.select(["run_id", "config", "overall_aggregate"])
        .join(baseline, on="run_id", how="left")
        .with_columns(
            (pl.col("overall_aggregate") - pl.col("baseline_aggregate"))
            .alias("delta_from_baseline")
        )
        .sort(["run_id", "config"])
    )


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
        p for p in session_dir.iterdir()
        if p.is_dir() and p.name != "variance_investigations"
    )

    if not run_dirs:
        print(f"No run directories found under {session_dir}. Exiting.")
        return

    run_ids = [p.name for p in run_dirs]
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

    variance_df = build_variance_df(comparison_df)
    write_variance_parquet(variance_df, session_dir)

if __name__ == "__main__":
    main()