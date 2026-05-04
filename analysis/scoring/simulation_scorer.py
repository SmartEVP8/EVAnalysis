"""
Scores are computed per snapshot by joining the
per-snapshot DataFrames from EVScores and StationScores. The run-wide
overall_aggregate is the mean of per-snapshot combined scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from datetime import datetime, timedelta

from helpers.constants import OUTPUT_ROOT
from analysis.scoring.ev_scorer import EVScores, EV_METRIC_WEIGHTS, compute_ev_scores
from analysis.scoring.station_scorer import StationScores, STATION_METRIC_WEIGHTS, compute_station_scores

GROUP_WEIGHTS: dict[str, int] = {
    "ev": 1,
    "station": 1,
}

TOTAL_GROUP_WEIGHT: int = sum(GROUP_WEIGHTS.values())

SIM_EPOCH = datetime(2024, 1, 1, 0, 0, 0)

def simtime_ms_to_label(simtime_ms: int) -> str:
    datetime = SIM_EPOCH + timedelta(milliseconds=simtime_ms)
    return datetime.strftime("%A %H:%M")


def compute_per_snapshot(
    ev_scores: EVScores,
    station_scores: StationScores,
) -> pl.DataFrame:
    ev_total_weight = sum(EV_METRIC_WEIGHTS.values())
    station_total_weight = sum(STATION_METRIC_WEIGHTS.values())

    ev_per_snapshot = ev_scores.per_snapshot.with_columns([
        (
            (
                EV_METRIC_WEIGHTS["path_deviation"] * pl.col("path_deviation_score")
                + EV_METRIC_WEIGHTS["delta_arrival"] * pl.col("delta_arrival_score")
                + EV_METRIC_WEIGHTS["ev_wait_time"] * pl.col("ev_wait_time_score")
                + EV_METRIC_WEIGHTS["missed_deadline"] * pl.col("missed_deadline_score")
            ) / ev_total_weight
        ).alias("ev_weighted_score")
    ])
    station_per_bucket = station_scores.per_bucket.with_columns([
        (
            (
                STATION_METRIC_WEIGHTS["utilization"] * pl.col("utilization_score")
                + STATION_METRIC_WEIGHTS["expected_wait_time"] * pl.col("expected_wait_score")
            ) / station_total_weight
        ).alias("station_weighted_score")
    ])

    return (
        ev_per_snapshot
        .join(station_per_bucket, on="simtime_ms", how="full", coalesce=True)
        .sort("simtime_ms")
        .with_columns([
            pl.col("ev_weighted_score").fill_null(0.0),
            pl.col("station_weighted_score").fill_null(0.0),
        ])
        .with_columns([
            (
                (
                    GROUP_WEIGHTS["ev"] * pl.col("ev_weighted_score")
                    + GROUP_WEIGHTS["station"] * pl.col("station_weighted_score")
                ) / TOTAL_GROUP_WEIGHT
            ).alias("combined_score")
        ])
    )


class SimulationScore:
    def __init__(
        self,
        run_id: str,
        source_path: str,
        ev_scores: EVScores,
        station_scores: StationScores,
    ) -> None:
        self.run_id = run_id
        self.source_path = source_path
        self.ev_scores = ev_scores
        self.station_scores = station_scores

        self.per_snapshot: pl.DataFrame = compute_per_snapshot(ev_scores, station_scores)

        self.overall_aggregate: float = self.per_snapshot["combined_score"].mean() or 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "simulation_config": {"source": self.source_path},
            "ev_scores": self.ev_scores.to_dict(),
            "station_scores": self.station_scores.to_dict(),
            "group_weights": GROUP_WEIGHTS,
            "per_snapshot_scores": [
                {
                    "time": simtime_ms_to_label(row["simtime_ms"]),
                    "simtime_ms": row["simtime_ms"],
                    "ev_weighted_score": row["ev_weighted_score"],
                    "station_weighted_score": row["station_weighted_score"],
                    "combined_score": row["combined_score"],
                }
                for row in self.per_snapshot.to_dicts()
            ],
            "overall_aggregate": round(self.overall_aggregate, 6),
        }

    def write_parquet(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
        df = self.per_snapshot.with_columns(
            pl.col("simtime_ms")
            .map_elements(simtime_ms_to_label, return_dtype=pl.String)
            .alias("time")
        ).select([
            "time",
            "simtime_ms",
            "ev_weighted_score",
            "station_weighted_score",
            "combined_score",
            "path_deviation_score",
            "delta_arrival_score",
            "ev_wait_time_score",
            "missed_deadline_score",
            "missed_proportion",
            "utilization_score",
            "expected_wait_score",
        ])
    
        df.write_parquet(output_path)
        print(f"[SimulationScorer] Wrote {output_path}")

        
    def write_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_id": self.run_id,
            "simulation_config": {"source": self.source_path},
            "ev_scores": self.ev_scores.to_dict(),
            "station_scores": self.station_scores.to_dict(),
            "group_weights": GROUP_WEIGHTS,
            "overall_aggregate": round(self.overall_aggregate, 6),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[SimulationScorer] Wrote {output_path}")


def compute_simulation_score(
    run_id: str,
    source_path: str,
    output_root: Path = OUTPUT_ROOT,
    output_path: Path | None = None,
) -> SimulationScore:
    """
    
    """
    if output_path is None:
        parquet_path = output_root / run_id / "simulation_score.parquet"
        json_path = output_root / run_id / "simulation_score.json"

    ev_scores = compute_ev_scores(run_id, output_root)
    station_scores = compute_station_scores(run_id, output_root)

    result = SimulationScore(
        run_id=run_id,
        source_path=source_path,
        ev_scores=ev_scores,
        station_scores=station_scores,
    )
    
    result.write_parquet(parquet_path)
    result.write_json(json_path)
    
    return result