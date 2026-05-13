from __future__ import annotations

import json
from pathlib import Path
from datetime import timedelta

import polars as pl

from helpers.constants import OUTPUT_ROOT
from analysis.scoring.scoring_config import ScoringConfig
from analysis.scoring.ev_scorer import EVScores, compute_ev_scores
from analysis.scoring.station_scorer import StationScores, compute_station_scores

def simtime_ms_to_label(simtime_ms: int, config: ScoringConfig) -> str:
    dt = config.sim_epoch + timedelta(milliseconds=simtime_ms)
    return dt.strftime("%A %H:%M")

def compute_per_snapshot(
    ev_scores: EVScores,
    station_scores: StationScores,
    config: ScoringConfig,
) -> pl.DataFrame:
    earliest_ev_time = ev_scores.per_snapshot["simtime_ms"].min()
    earliest_station_time = station_scores.per_bucket["simtime_ms"].min()

    if earliest_ev_time is None or earliest_station_time is None:
        raise ValueError("Cannot compute warmup cutoff: one or both score DataFrames are empty.")

    sim_start_ms = min(earliest_ev_time, earliest_station_time)
    warmup_cutoff_ms = sim_start_ms + config.warmup_ms

    ev_per_snapshot = ev_scores.per_snapshot.filter(pl.col("simtime_ms") > warmup_cutoff_ms).with_columns([
        (
            (
                config.ev_metric_weights["path_deviation"] * pl.col("path_deviation_score")
                + config.ev_metric_weights["delta_arrival"] * pl.col("delta_arrival_score")
                + config.ev_metric_weights["ev_wait_time"] * pl.col("ev_wait_time_score")
                + config.ev_metric_weights["missed_deadline"] * pl.col("missed_deadline_score")
            ) / config.ev_total_weight
        ).alias("ev_weighted_score")
    ])

    station_per_bucket = station_scores.per_bucket.filter(pl.col("simtime_ms") > warmup_cutoff_ms).with_columns([
        (
            (
                config.station_metric_weights["utilization"] * pl.col("utilization_score")
                + config.station_metric_weights["expected_wait_time"] * pl.col("expected_wait_score")
            ) / config.station_total_weight
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
                    config.group_weights["ev"] * pl.col("ev_weighted_score")
                    + config.group_weights["station"] * pl.col("station_weighted_score")
                ) / config.total_group_weight
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
        config: ScoringConfig,
    ) -> None:
        self.run_id = run_id
        self.source_path = source_path
        self.ev_scores = ev_scores
        self.station_scores = station_scores
        self.config = config

        self.per_snapshot: pl.DataFrame = compute_per_snapshot(ev_scores, station_scores, config)

        self.path_deviation_aggregate: float = self.per_snapshot["path_deviation_score"].mean() or 0.0
        self.delta_arrival_aggregate: float = self.per_snapshot["delta_arrival_score"].mean() or 0.0
        self.ev_wait_time_aggregate: float = self.per_snapshot["ev_wait_time_score"].mean() or 0.0
        self.missed_deadline_aggregate: float = self.per_snapshot["missed_deadline_score"].mean() or 0.0

        self.ev_weighted_aggregate: float = (
            config.ev_metric_weights["path_deviation"] * self.path_deviation_aggregate
            + config.ev_metric_weights["delta_arrival"] * self.delta_arrival_aggregate
            + config.ev_metric_weights["ev_wait_time"] * self.ev_wait_time_aggregate
            + config.ev_metric_weights["missed_deadline"] * self.missed_deadline_aggregate
        ) / config.ev_total_weight

        self.utilization_aggregate: float = self.per_snapshot["utilization_score"].mean() or 0.0
        self.expected_wait_aggregate: float = self.per_snapshot["expected_wait_score"].mean() or 0.0

        self.station_weighted_aggregate: float = (
            config.station_metric_weights["utilization"] * self.utilization_aggregate
            + config.station_metric_weights["expected_wait_time"] * self.expected_wait_aggregate
        ) / config.station_total_weight

        self.overall_aggregate: float = self.per_snapshot["combined_score"].mean() or 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "simulation_config": {"source": self.source_path},
            "ev_scores": {
                "per_metric": {
                    "path_deviation_minutes": {
                        "higher_is_better": False,
                        "bucket_labels": self.config.path_deviation_labels,
                        "bucket_weights": [w for _, w in self.config.path_deviation_buckets],
                        "aggregate_score": round(self.path_deviation_aggregate, 6),
                    },
                    "delta_arrival_minutes": {
                        "higher_is_better": False,
                        "bucket_labels": self.config.delta_arrival_labels,
                        "bucket_weights": [w for _, w in self.config.delta_arrival_buckets],
                        "aggregate_score": round(self.delta_arrival_aggregate, 6),
                    },
                    "ev_wait_time": {
                        "higher_is_better": False,
                        "aggregate_score": round(self.ev_wait_time_aggregate, 6),
                    },
                    "missed_deadline": {
                        "higher_is_better": False,
                        "aggregate_score": round(self.missed_deadline_aggregate, 6),
                    },
                },
                "metric_weights": self.config.ev_metric_weights,
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
                        "aggregate_score": round(self.expected_wait_aggregate, 6),
                    },
                },
                "metric_weights": self.config.station_metric_weights,
                "weighted_aggregate": round(self.station_weighted_aggregate, 6),
            },
            "group_weights": self.config.group_weights,
            "per_snapshot_scores": [
                {
                    "time": simtime_ms_to_label(row["simtime_ms"], self.config),
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
            .map_elements(lambda ms: simtime_ms_to_label(ms, self.config), return_dtype=pl.String)
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
            "group_weights": self.config.group_weights,
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
    config: ScoringConfig | None = None,
) -> SimulationScore:
    if config is None:
        config = ScoringConfig()

    if output_path is None:
        parquet_path = output_root / run_id / "simulation_score.parquet"
        json_path = output_root / run_id / "simulation_score.json"
    else:
        parquet_path = output_path
        json_path = output_path.with_suffix(".json")

    ev_scores = compute_ev_scores(run_id, output_root, config)
    station_scores = compute_station_scores(run_id, output_root, config)

    result = SimulationScore(
        run_id=run_id,
        source_path=source_path,
        ev_scores=ev_scores,
        station_scores=station_scores,
        config=config,
    )

    result.write_parquet(parquet_path)
    result.write_json(json_path)

    return result
