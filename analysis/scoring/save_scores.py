"""
Module: save_scores
Assembles and persists the three scoring JSON files for a simulation run.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from analysis.scoring.station_scorer import score_stations
from analysis.scoring.ev_scorer import score_evs
from visualisation.dashboards.scoring_dashboard import generate_dashboard


def run_scoring(
    run_id: str,
    station_snapshots: pl.DataFrame,
    ev_percentiles: pl.DataFrame,
    simulation_config: dict,
    output_root: Path = Path("runs"),
    time_aggregation: str = "max",
) -> None:
    scoring_dir = output_root / run_id / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)

    station_scores = score_stations(station_snapshots, time_aggregation)
    ev_scores = score_evs(ev_percentiles, time_aggregation)

    overall = round(
        ((station_scores["aggregate"] + ev_scores["aggregate"]) / 2.0) * 100, 2
    )

    write_json(
        {
            "run_id": run_id,
            "simulation_config": simulation_config,
            "station_scores": station_scores,
            "ev_scores": ev_scores,
            "overall_aggregate": overall,
        },
        scoring_dir / "simulation_score.json",
    )

    print(f"[Scoring] Written to {scoring_dir}")
    print(f"  Station aggregate : {station_scores['aggregate']:.2f}")
    print(f"  EV aggregate : {ev_scores['aggregate']:.2f}")
    print(f"  Overall : {overall:.2f}")

    generate_dashboard(run_id, overall, ev_scores, station_scores, scoring_dir)


def write_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_serialiser)


def json_serialiser(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")