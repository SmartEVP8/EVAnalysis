"""
Module: save_scores
Assembles and persists the three scoring JSON files for a simulation run.

Output structure:
  runs/{run_id}/scoring/station_score.json
  runs/{run_id}/scoring/ev_score.json
  runs/{run_id}/scoring/simulation_score.json
"""

import json
from pathlib import Path

import polars as pl

from analysis.scoring.station_scorer import score_stations
from analysis.scoring.ev_scorer import score_evs


def run_scoring(
    run_id: str,
    station_snapshots: pl.DataFrame,
    station_percentiles: pl.DataFrame,
    wait_time_metrics: pl.DataFrame,
    arrival_snapshots: pl.DataFrame,
    arrival_percentiles: pl.DataFrame,
    simulation_config: dict,
    output_root: Path = Path("runs"),
) -> None:
    """
    Computes all scores and writes three JSON files to runs/{run_id}/scoring/.

    Args:
        run_id:               The simulation run identifier.
        station_snapshots:    DataFrame from station_snapshots.parquet
        station_percentiles:  DataFrame from station_percentiles.parquet
        wait_time_metrics:    DataFrame from WaitTimeInQueueMetric.parquet
        arrival_snapshots:    DataFrame from arrival_snapshots.parquet
        arrival_percentiles:  DataFrame from arrival_percentiles.parquet
        simulation_config:    Dict of simulation configuration values to embed in the summary.
        output_root:          Root directory for run outputs (default: "runs/")
    """

    scoring_dir = output_root / run_id / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)

    # --- Compute scores ---
    station_scores = score_stations(station_snapshots, station_percentiles, wait_time_metrics)
    ev_scores      = score_evs(arrival_snapshots, arrival_percentiles)

    # --- station_score.json ---
    _write_json(
        {"run_id": run_id, "scores": station_scores},
        scoring_dir / "station_score.json",
    )

    # --- ev_score.json ---
    _write_json(
        {"run_id": run_id, "scores": ev_scores},
        scoring_dir / "ev_score.json",
    )

    # --- simulation_score.json ---
    _write_json(
        {
            "run_id":            run_id,
            "simulation_config": simulation_config,
            "station_scores":    station_scores,
            "ev_scores":         ev_scores,
        },
        scoring_dir / "simulation_score.json",
    )

    print(f"[Scoring] Written to {scoring_dir}")


def _write_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_serialiser)


def _json_serialiser(obj):
    """Handles types that json.dump can't serialise by default (e.g. numpy/polars scalars)."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")