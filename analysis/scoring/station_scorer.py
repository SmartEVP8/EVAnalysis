"""
Scores station-level simulation metrics per snapshot bucket.

Per-bucket metrics
utilization_score : mean normalised utilization across stations (relative to bucket max)
expected_wait_score : mean Gaussian decay score for expected wait time

The run-wide aggregate for each metric is the mean of its per-bucket scores.
weighted_aggregate is the weighted mean of the two metric aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from helpers.constants import PERCENTILES

import polars as pl


PERCENTILE_NAMES:  list[str]   = ["p25", "p50", "p75", "p90", "p95", "p99"]

METRIC_WEIGHTS: dict[str, int] = {
    "utilization":        1,
    "expected_wait_time": 3,
}

WAIT_DECAY_MINUTES: float = 45.0


def wait_score_expr(wait_col: str) -> pl.Expr:
    """
    Gaussian decay score for wait time: exp(-(x/WAIT_DECAY_MINUTES)²), averaged per group.

    A wait of 0 minutes scores 1.0; longer waits score progressively lower,
    with 45 minutes scoring roughly 0.37, and waits over 90 minutes scoring near 0.

    exp( -(45 / 45)² ) = exp(-1) ≈ 0.3679
    exp( -(90 / 45)² ) = exp(-4) ≈ 0.0183
    """
    x = pl.col(wait_col)
    return (-((x / WAIT_DECAY_MINUTES) ** 2)).exp()


def compute_station_scores(run_id: str, output_root: Path) -> StationScores:
    """
    Computes per-bucket and aggregate station scores for a simulation run.

    Reads station_snapshots.parquet, computes utilization and wait-time scores
    per bucket, and returns a StationScores dataclass.
    """
    snapshots_path = output_root / run_id / "analysis" / "station_snapshots.parquet"
    snapshots = pl.read_parquet(snapshots_path)

    wait_col = "expected_wait_minutes"
    if wait_col not in snapshots.columns:
        raise ValueError(
            f"Expected column '{wait_col}' not found in {snapshots_path}"
        )

    snapshots = snapshots.with_columns(
        wait_score_expr(wait_col).alias("wait_score")
    )

    per_bucket = (
        snapshots
        .group_by("simtime_ms")
        .agg([
            (
                pl.when(pl.col("utilization").max() > 0)
                  .then(pl.col("utilization") / pl.col("utilization").max())
                  .otherwise(0.0)
            ).mean().alias("utilization_score"),

            pl.col("wait_score").mean().alias("expected_wait_score"),

            *[
                pl.col("wait_score").quantile(p).alias(f"wait_score_{name}")
                for p, name in zip(PERCENTILES, PERCENTILE_NAMES)
            ],
        ])
        .fill_nan(0.0)
        .sort("simtime_ms")
    )

    utilization_aggregate: float = per_bucket["utilization_score"].mean()
    expected_wait_aggregate: float = per_bucket["expected_wait_score"].mean()

    total_weight: int = sum(METRIC_WEIGHTS.values())
    weighted_aggregate: float = (
        METRIC_WEIGHTS["utilization"] * utilization_aggregate
        + METRIC_WEIGHTS["expected_wait_time"] * expected_wait_aggregate
    ) / total_weight

    return StationScores(
        per_bucket=per_bucket,
        utilization_aggregate=utilization_aggregate,
        expected_wait_time_aggregate=expected_wait_aggregate,
        weighted_aggregate=weighted_aggregate,
        number_of_stations=snapshots["StationId"].n_unique(),
    )



@dataclass
class StationScores:
    per_bucket: pl.DataFrame

    utilization_aggregate: float
    expected_wait_time_aggregate: float
    weighted_aggregate: float
    number_of_stations: int

    def to_dict(self) -> dict:
        return {
            "per_metric": {
                "utilization": {
                    "higher_is_better": True,
                    "aggregate_score": round(self.utilization_aggregate, 6),
                },
                "expected_wait_time": {
                    "higher_is_better": True,
                    "aggregate_score": round(self.expected_wait_time_aggregate, 6),
                },
            },
            "number_of_stations": self.number_of_stations,
            "metric_weights": METRIC_WEIGHTS,
            "weighted_aggregate": round(self.weighted_aggregate, 6),
        }