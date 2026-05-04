"""
Scores EV-level simulation metrics per snapshot bucket.

Per-bucket metrics
path_deviation  : time-bucket weighted penalty for route deviation
delta_arrival   : time-bucket weighted penalty for late arrival
ev_wait_time    : Gaussian decay score for queue wait time
missed_deadline : proportion of non-direct-drive arrivals that missed

The run-wide aggregate for each metric is the mean of its per-bucket scores.
weighted_aggregate is the weighted mean of the four metric aggregates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from helpers.io_helpers import infer_snapshot_interval_ms
from helpers.constants import PERCENTILES

PATH_DEVIATION_BUCKETS: list[tuple[float, int]] = [
    (5,            1),
    (10,           1),
    (15,           1),
    (30,           1),
    (60,           2),
    (float("inf"), 3),
]
PATH_DEVIATION_BUCKET_LABELS: list[str] = ["5", "10", "15", "30", "60", "60+"]

DELTA_ARRIVAL_BUCKETS: list[tuple[float, int]] = [
    (0,            2),
    (5,            1),
    (10,           1),
    (15,           1),
    (30,           1),
    (60,           1),
    (float("inf"), 1),
]
DELTA_ARRIVAL_BUCKET_LABELS: list[str] = ["0", "5", "10", "15", "30", "60", "60+"]

PERCENTILE_NAMES:  list[str] = ["p25", "p50", "p75", "p90", "p95", "p99"]

EV_METRIC_WEIGHTS = {
    "path_deviation": 1,
    "delta_arrival": 1,
    "ev_wait_time": 3,
    "missed_deadline": 2,
}

WAIT_DECAY_MINUTES: float = 45.0


def bucket_score_expr(col: str, buckets: list[tuple[float, int]]) -> pl.Expr:
    """
    Builds a Polars expression that computes the normalised bucket-penalty score
    for *col* within a group.

    Each row is assigned the weight of the bucket it falls into. The group score
    is mean(row_weight) / total_weight, so the score ranges between 0 and 1.
    """
    total_weight: int = sum(weight for _, weight in buckets)

    previous_upper = float("-inf")
    weight_expr: pl.Expr = pl.lit(None, dtype=pl.Float64)

    for upper_bound, weight in buckets:
        if upper_bound == 0:
            bucket_filter = pl.col(col) == 0.0
        elif math.isinf(upper_bound):
            bucket_filter = pl.col(col) > previous_upper
        else:
            bucket_filter = (pl.col(col) > previous_upper) & (pl.col(col) <= upper_bound)

        weight_expr = pl.when(bucket_filter).then(pl.lit(float(weight))).otherwise(weight_expr)
        previous_upper = upper_bound

    return (weight_expr.mean() / total_weight).alias(f"{col}_score")


def wait_score_expr() -> pl.Expr:
    """
    Gaussian decay score for wait time: exp(-(x/WAIT_DECAY_MINUTES)²), averaged per group.

    A wait of 0 minutes scores 1.0; longer waits score progressively lower,
    with 45 minutes scoring roughly 0.37, and waits over 90 minutes scoring near 0.

    exp( -(45 / 45)² ) = exp(-1) ≈ 0.3679
    exp( -(90 / 45)² ) = exp(-4) ≈ 0.0183
    """
    wait_time_score = pl.col("wait_minutes")
    return ((-((wait_time_score / WAIT_DECAY_MINUTES) ** 2)).exp()).mean().alias("ev_wait_time_score")


def missed_deadline_exprs() -> list[pl.Expr]:
    """
    Returns aggregation expressions for missed-deadline statistics per group.
    """
    total = pl.len()
    direct = pl.col("drive_directly").sum()
    missed = pl.col("missed_deadline").sum()
    not_direct = total - direct
    proportion = pl.when(not_direct > 0).then(missed / not_direct).otherwise(0.0)

    return [
        proportion.alias("missed_proportion"),
        (1.0 - proportion).alias("missed_deadline_score"),
        total.alias("total_arrivals"),
        direct.alias("direct_drive_arrivals"),
        missed.alias("missed_deadlines"),
    ]



def compute_ev_scores(run_id: str, output_root: Path) -> EVScores:
    """
    Computes per-bucket and aggregate EV scores for a simulation run.

    Reads arrival_snapshots.parquet and waittime_snapshots.parquet, aligns
    them onto a shared bucket grid, and returns an EVScores dataclass.

    Args:
        run_id: Simulation run identifier (e.g. 'Run_001').
        output_root: Root directory containing all run output.

    Returns:
        EVScores with per-bucket DataFrame and run-wide aggregates.
    """
    base_dir = output_root / run_id / "analysis"

    arrivals = pl.read_parquet(base_dir / "arrival_snapshots.parquet")
    wait_time_df = pl.read_parquet(base_dir / "waittime_snapshots.parquet")

    snapshot_interval_ms = infer_snapshot_interval_ms(
        output_root / run_id / "analysis" / "station_snapshots.parquet"
    )

    arrival_scores = (
        arrivals
        .with_columns([
            (
                (pl.col("simtime_ms") // snapshot_interval_ms) * snapshot_interval_ms
            ).alias("simtime_ms")
        ])
        .group_by("simtime_ms")
        .agg([
            bucket_score_expr("path_deviation_minutes", PATH_DEVIATION_BUCKETS),
            bucket_score_expr("delta_arrival_minutes",  DELTA_ARRIVAL_BUCKETS),
            *missed_deadline_exprs(),
        ])
    )

    wait_scores = (
        wait_time_df
        .group_by("simtime_ms")
        .agg([
            wait_score_expr().alias("ev_wait_time_score"),

            *[
                pl.col("wait_minutes").quantile(percentile).alias(f"wait_minutes_{name}")
                for percentile, name in zip(PERCENTILES, PERCENTILE_NAMES)
            ],
        ])
    )

    quantile_cols = [f"wait_minutes_{name}" for name in PERCENTILE_NAMES]

    per_snapshot = (
        arrival_scores
        .join(wait_scores, on="simtime_ms", how="left")
        .with_columns(pl.col("ev_wait_time_score").fill_null(1.0))
        .sort("simtime_ms")
        .rename({
            "path_deviation_minutes_score": "path_deviation_score",
            "delta_arrival_minutes_score":  "delta_arrival_score",
        })
        .select([
            "simtime_ms",
            "path_deviation_score",
            "delta_arrival_score",
            "ev_wait_time_score",
            "missed_deadline_score",
            "missed_proportion",
            "total_arrivals",
            "direct_drive_arrivals",
            "missed_deadlines",
            *quantile_cols,
        ])
    )

    path_deviation_aggregate: float = per_snapshot["path_deviation_score"].mean()  or 0.0
    delta_arrival_aggregate: float = per_snapshot["delta_arrival_score"].mean()   or 0.0
    ev_wait_time_aggregate: float = per_snapshot["ev_wait_time_score"].mean()    or 0.0
    missed_deadline_aggregate: float = per_snapshot["missed_deadline_score"].mean() or 0.0

    total_weight: int = sum(EV_METRIC_WEIGHTS.values())
    weighted_aggregate: float = (
        EV_METRIC_WEIGHTS["path_deviation"] * path_deviation_aggregate
        + EV_METRIC_WEIGHTS["delta_arrival"] * delta_arrival_aggregate
        + EV_METRIC_WEIGHTS["ev_wait_time"] * ev_wait_time_aggregate
        + EV_METRIC_WEIGHTS["missed_deadline"] * missed_deadline_aggregate
    ) / total_weight

    return EVScores(
        per_snapshot=per_snapshot,
        path_deviation_aggregate=path_deviation_aggregate,
        delta_arrival_aggregate=delta_arrival_aggregate,
        ev_wait_time_aggregate=ev_wait_time_aggregate,
        missed_deadline_aggregate=missed_deadline_aggregate,
        weighted_aggregate=weighted_aggregate,
    )


@dataclass
class EVScores:
    per_snapshot: pl.DataFrame

    path_deviation_aggregate: float
    delta_arrival_aggregate: float
    ev_wait_time_aggregate: float
    missed_deadline_aggregate: float
    weighted_aggregate: float

    def to_dict(self) -> dict:
        return {
            "per_metric": {
                "path_deviation_minutes": {
                    "higher_is_better": False,
                    "bucket_labels": PATH_DEVIATION_BUCKET_LABELS,
                    "bucket_weights": [weight for _, weight in PATH_DEVIATION_BUCKETS],
                    "aggregate_score": round(self.path_deviation_aggregate, 6),
                },
                "delta_arrival_minutes": {
                    "higher_is_better": False,
                    "bucket_labels": DELTA_ARRIVAL_BUCKET_LABELS,
                    "bucket_weights": [weight for _, weight in DELTA_ARRIVAL_BUCKETS],
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
            "metric_weights": EV_METRIC_WEIGHTS,
            "weighted_aggregate": round(self.weighted_aggregate, 6),
        }