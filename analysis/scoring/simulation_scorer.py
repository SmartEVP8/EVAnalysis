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
from analysis.scoring.default_scores import (
    DEFAULT_SCORING_CONFIG,
    ScoringConfig,
    bucket_labels, 
)
from analysis.scoring.ev_scorer import EVScores, compute_ev_scores
from analysis.scoring.station_scorer import StationScores, compute_station_scores

SIM_EPOCH = datetime(2024, 1, 1, 0, 0, 0)

WARMUP_MS: int = (3 * 60 * 60 * 1000) - 1200000  # 3 hours minus 20 minutes to exclude the initial ramp-up and ramp-down periods


def simtime_ms_to_label(simtime_ms: int) -> str:
    dt = SIM_EPOCH + timedelta(milliseconds=simtime_ms)
    return dt.strftime("%A %H:%M")


def compute_per_snapshot(
    ev_scores: EVScores,
    station_scores: StationScores,
    *,
    ev_metric_weights: dict[str, int] = DEFAULT_SCORING_CONFIG.ev_metric_weights,
    station_metric_weights: dict[str, int] = DEFAULT_SCORING_CONFIG.station_metric_weights,
    group_weights: dict[str, int] = DEFAULT_SCORING_CONFIG.group_weights,
) -> pl.DataFrame:
    ev_total_weight = sum(ev_metric_weights.values())
    station_total_weight = sum(station_metric_weights.values())

    earliest_ev_time = ev_scores.per_snapshot["simtime_ms"].min()
    earliest_station_time = station_scores.per_bucket["simtime_ms"].min()

    if earliest_ev_time is None or earliest_station_time is None:
        raise ValueError("Cannot compute warmup cutoff: one or both score DataFrames are empty.")

    sim_start_ms = min(earliest_ev_time, earliest_station_time)
    warmup_cutoff_ms = sim_start_ms + WARMUP_MS

    ev_per_snapshot = ev_scores.per_snapshot.filter(pl.col("simtime_ms") > warmup_cutoff_ms).with_columns([
        (
            (
                ev_metric_weights["path_deviation"] * pl.col("path_deviation_score")
                + ev_metric_weights["delta_arrival"] * pl.col("delta_arrival_score")
                + ev_metric_weights["ev_wait_time"] * pl.col("ev_wait_time_score")
                + ev_metric_weights["missed_deadline"] * pl.col("missed_deadline_score")
            ) / ev_total_weight
        ).alias("ev_weighted_score")
    ])

    station_per_bucket = station_scores.per_bucket.filter(pl.col("simtime_ms") > warmup_cutoff_ms).with_columns([
        (
            (
                station_metric_weights["utilization"] * pl.col("utilization_score")
                + station_metric_weights["expected_wait_time"] * pl.col("expected_wait_score")
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
                    group_weights["ev"] * pl.col("ev_weighted_score")
                    + group_weights["station"] * pl.col("station_weighted_score")
                ) / sum(group_weights.values())
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
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        self.run_id = run_id
        self.source_path = source_path
        self.ev_scores = ev_scores
        self.station_scores = station_scores
        self.scoring_config = scoring_config or DEFAULT_SCORING_CONFIG

        self.per_snapshot: pl.DataFrame = compute_per_snapshot(
            ev_scores,
            station_scores,
            ev_metric_weights=self.scoring_config.ev_metric_weights,
            station_metric_weights=self.scoring_config.station_metric_weights,
            group_weights=self.scoring_config.group_weights,
        )

        # EV metric aggregates
        self.path_deviation_aggregate: float = self.per_snapshot["path_deviation_score"].mean() or 0.0
        self.delta_arrival_aggregate: float = self.per_snapshot["delta_arrival_score"].mean() or 0.0
        self.ev_wait_time_aggregate: float = self.per_snapshot["ev_wait_time_score"].mean() or 0.0
        self.missed_deadline_aggregate: float = self.per_snapshot["missed_deadline_score"].mean() or 0.0

        ev_total_weight = sum(self.scoring_config.ev_metric_weights.values())
        self.ev_weighted_aggregate: float = (
            self.scoring_config.ev_metric_weights["path_deviation"] * self.path_deviation_aggregate
            + self.scoring_config.ev_metric_weights["delta_arrival"] * self.delta_arrival_aggregate
            + self.scoring_config.ev_metric_weights["ev_wait_time"] * self.ev_wait_time_aggregate
            + self.scoring_config.ev_metric_weights["missed_deadline"] * self.missed_deadline_aggregate
        ) / ev_total_weight

        # Station metric aggregates
        self.utilization_aggregate: float = self.per_snapshot["utilization_score"].mean() or 0.0
        self.expected_wait_aggregate: float = self.per_snapshot["expected_wait_score"].mean() or 0.0

        station_total_weight = sum(self.scoring_config.station_metric_weights.values())
        self.station_weighted_aggregate: float = (
            self.scoring_config.station_metric_weights["utilization"] * self.utilization_aggregate
            + self.scoring_config.station_metric_weights["expected_wait_time"] * self.expected_wait_aggregate
        ) / station_total_weight

        # Overall
        self.overall_aggregate: float = self.per_snapshot["combined_score"].mean() or 0.0

        if len(self.per_snapshot) > 0:
            max_idx = self.per_snapshot["combined_score"].arg_max()
            min_idx = self.per_snapshot["combined_score"].arg_min()
            max_row = self.per_snapshot[max_idx]
            min_row = self.per_snapshot[min_idx]

            self.highest_score = float(max_row["combined_score"][0])
            self.highest_score_time = simtime_ms_to_label(max_row["simtime_ms"][0])
            self.highest_score_simtime_ms = int(max_row["simtime_ms"][0])

            self.lowest_score = float(min_row["combined_score"][0])
            self.lowest_score_time = simtime_ms_to_label(min_row["simtime_ms"][0])
            self.lowest_score_simtime_ms = int(min_row["simtime_ms"][0])
        else:
            self.highest_score = 0.0
            self.highest_score_time = "N/A"
            self.highest_score_simtime_ms = 0
            self.lowest_score = 0.0
            self.lowest_score_time = "N/A"
            self.lowest_score_simtime_ms = 0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "simulation_config": {"source": self.source_path},
            "ev_scores": {
                "per_metric": {
                    "path_deviation_minutes": {
                        "higher_is_better": False,
                        "bucket_labels": bucket_labels(self.scoring_config.path_deviation_buckets),
                        "bucket_weights": [w for _, w in self.scoring_config.path_deviation_buckets],
                        "aggregate_score": round(self.path_deviation_aggregate, 6),
                    },
                    "delta_arrival_minutes": {
                        "higher_is_better": False,
                        "bucket_labels": bucket_labels(self.scoring_config.delta_arrival_buckets),
                        "bucket_weights": [w for _, w in self.scoring_config.delta_arrival_buckets],
                        "aggregate_score": round(self.delta_arrival_aggregate, 6),
                    },
                    "ev_wait_time": {
                        "higher_is_better": False,
                        "percentile_labels": [name for name, _ in self.scoring_config.wait_time_buckets],
                        "percentile_weights": [w for _, w in self.scoring_config.wait_time_buckets],
                        "aggregate_score": round(self.ev_wait_time_aggregate, 6),
                    },
                    "missed_deadline": {
                        "higher_is_better": False,
                        "aggregate_score": round(self.missed_deadline_aggregate, 6),
                    },
                },
                "metric_weights": self.scoring_config.ev_metric_weights,
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
                        "percentile_labels": [name for name, _ in self.scoring_config.expected_wait_time_buckets],
                        "percentile_weights": [w for _, w in self.scoring_config.expected_wait_time_buckets],
                        "aggregate_score": round(self.expected_wait_aggregate, 6),
                    },
                },
                "metric_weights": self.scoring_config.station_metric_weights,
                "weighted_aggregate": round(self.station_weighted_aggregate, 6),
            },
            "group_weights": self.scoring_config.group_weights,
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
            "group_weights": self.scoring_config.group_weights,
            "highest_score": {
                "score": round(self.highest_score, 2),
                "time": self.highest_score_time,
                "simtime_ms": self.highest_score_simtime_ms,
            },
            "lowest_score": {
                "score": round(self.lowest_score, 2),
                "time": self.lowest_score_time,
                "simtime_ms": self.lowest_score_simtime_ms,
            },
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
    scoring_config: ScoringConfig | None = None,
) -> SimulationScore:
    if output_path is None:
        parquet_path = output_root / run_id / "simulation_score.parquet"
        json_path = output_root / run_id / "simulation_score.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path = output_path.with_suffix(".parquet")
        json_path = output_path.with_suffix(".json")

    scoring_config = scoring_config or DEFAULT_SCORING_CONFIG

    ev_scores = compute_ev_scores(
        run_id=run_id,
        output_root=output_root,
        path_deviation_buckets=scoring_config.path_deviation_buckets,
        delta_arrival_buckets=scoring_config.delta_arrival_buckets,
        wait_time_buckets=scoring_config.wait_time_buckets,
    )
    station_scores = compute_station_scores(
        run_id=run_id,
        output_root=output_root,
        expected_wait_time_buckets=scoring_config.expected_wait_time_buckets,
    )

    result = SimulationScore(
        run_id=run_id,
        source_path=source_path,
        ev_scores=ev_scores,
        station_scores=station_scores,
        scoring_config=scoring_config,
    )

    result.write_parquet(parquet_path)
    result.write_json(json_path)

    return result