"""
Scores are computed per snapshot by joining the per-snapshot DataFrames from
EVScores and StationScores. All metric aggregates and the run-wide
overall_aggregate are computed here from those per-snapshot scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl

from helpers.constants import OUTPUT_ROOT
from helpers.scoring_weights import (
    DELTA_ARRIVAL_BUCKET_LABELS,
    DELTA_ARRIVAL_BUCKETS,
    EXPECTED_WAIT_TIME_BUCKETS,
    EV_METRIC_WEIGHTS,
    GROUP_WEIGHTS,
    PATH_DEVIATION_BUCKET_LABELS,
    PATH_DEVIATION_BUCKETS,
    STATION_METRIC_WEIGHTS,
    WAIT_TIME_BUCKETS,
    WARMUP_MS,
)
from analysis.scoring.ev_scorer import (
    EVScores,
    compute_ev_scores,
)
from analysis.scoring.station_scorer import (
    StationScores,
    compute_station_scores,
)

TOTAL_GROUP_WEIGHT: int = sum(GROUP_WEIGHTS.values())

SIM_EPOCH = datetime(2024, 1, 1, 0, 0, 0)


def simtime_ms_to_label(simtime_ms: int) -> str:
    dt = SIM_EPOCH + timedelta(milliseconds=simtime_ms)
    return dt.strftime("%A %H:%M")


def compute_per_snapshot(
    ev_scores: EVScores,
    station_scores: StationScores,
) -> pl.DataFrame:
    ev_total_weight = sum(EV_METRIC_WEIGHTS.values())
    station_total_weight = sum(STATION_METRIC_WEIGHTS.values())
    
    earliest_ev_time = ev_scores.per_snapshot["simtime_ms"].min()
    earliest_station_time = station_scores.per_bucket["simtime_ms"].min()

    if earliest_ev_time is None or earliest_station_time is None:
        raise ValueError("Cannot compute warmup cutoff: one or both score DataFrames are empty.")

    sim_start_ms = min(earliest_ev_time, earliest_station_time)
    warmup_cutoff_ms = sim_start_ms + WARMUP_MS

    ev_per_snapshot = ev_scores.per_snapshot.filter(pl.col("simtime_ms") > warmup_cutoff_ms).with_columns([
        (
            (
                EV_METRIC_WEIGHTS["path_deviation"] * pl.col("path_deviation_score")
                + EV_METRIC_WEIGHTS["delta_arrival"] * pl.col("delta_arrival_score")
                + EV_METRIC_WEIGHTS["ev_wait_time"] * pl.col("ev_wait_time_score")
                + EV_METRIC_WEIGHTS["missed_deadline"] * pl.col("missed_deadline_score")
            ) / ev_total_weight
        ).alias("ev_weighted_score")
    ])

    station_per_bucket = station_scores.per_bucket.filter(pl.col("simtime_ms") > warmup_cutoff_ms).with_columns([
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

        # EV metric aggregates
        self.path_deviation_aggregate: float = self.per_snapshot["path_deviation_score"].mean() or 0.0
        self.delta_arrival_aggregate: float = self.per_snapshot["delta_arrival_score"].mean() or 0.0
        self.ev_wait_time_aggregate: float = self.per_snapshot["ev_wait_time_score"].mean() or 0.0
        self.missed_deadline_aggregate: float = self.per_snapshot["missed_deadline_score"].mean() or 0.0

        ev_total_weight = sum(EV_METRIC_WEIGHTS.values())
        self.ev_weighted_aggregate: float = (
            EV_METRIC_WEIGHTS["path_deviation"] * self.path_deviation_aggregate
            + EV_METRIC_WEIGHTS["delta_arrival"] * self.delta_arrival_aggregate
            + EV_METRIC_WEIGHTS["ev_wait_time"] * self.ev_wait_time_aggregate
            + EV_METRIC_WEIGHTS["missed_deadline"] * self.missed_deadline_aggregate
        ) / ev_total_weight

        # Station metric aggregates
        self.utilization_aggregate: float = self.per_snapshot["utilization_score"].mean() or 0.0
        self.expected_wait_aggregate: float = self.per_snapshot["expected_wait_score"].mean() or 0.0

        station_total_weight = sum(STATION_METRIC_WEIGHTS.values())
        self.station_weighted_aggregate: float = (
            STATION_METRIC_WEIGHTS["utilization"] * self.utilization_aggregate
            + STATION_METRIC_WEIGHTS["expected_wait_time"] * self.expected_wait_aggregate
        ) / station_total_weight

        # Overall
        self.overall_aggregate: float = self.per_snapshot["combined_score"].mean() or 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "simulation_config": {"source": self.source_path},
            "ev_scores": {
                "per_metric": {
                    "path_deviation_minutes": {
                        "higher_is_better": False,
                        "bucket_labels": PATH_DEVIATION_BUCKET_LABELS,
                        "bucket_weights": [w for _, w in PATH_DEVIATION_BUCKETS],
                        "aggregate_score": round(self.path_deviation_aggregate, 6),
                    },
                    "delta_arrival_minutes": {
                        "higher_is_better": False,
                        "bucket_labels": DELTA_ARRIVAL_BUCKET_LABELS,
                        "bucket_weights": [w for _, w in DELTA_ARRIVAL_BUCKETS],
                        "aggregate_score": round(self.delta_arrival_aggregate, 6),
                    },
                    "ev_wait_time": {
                        "higher_is_better": False,
                        "percentile_labels": [name for name, _ in WAIT_TIME_BUCKETS],
                        "percentile_weights": [w for _, w in WAIT_TIME_BUCKETS],
                        "aggregate_score": round(self.ev_wait_time_aggregate, 6),
                    },
                    "missed_deadline": {
                        "higher_is_better": False,
                        "aggregate_score": round(self.missed_deadline_aggregate, 6),
                    },
                },
                "metric_weights": EV_METRIC_WEIGHTS,
                "weighted_aggregate": round(self.ev_weighted_aggregate, 6),
            },
            "station_scores": {
                "per_metric": {
                    "utilization": {
                        "higher_is_better": True,
                        "aggregate_score": round(self.utilization_aggregate, 6),
                    },
                    "expected_wait_time": {
                        "higher_is_better": True,
                        "percentile_labels": [name for name, _ in EXPECTED_WAIT_TIME_BUCKETS],
                        "percentile_weights": [w for _, w in EXPECTED_WAIT_TIME_BUCKETS],
                        "aggregate_score": round(self.expected_wait_aggregate, 6),
                    },
                },
                "metric_weights": STATION_METRIC_WEIGHTS,
                "weighted_aggregate": round(self.station_weighted_aggregate, 6),
            },
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
            "ev_scores": self.to_dict()["ev_scores"],
            "station_scores": self.to_dict()["station_scores"],
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