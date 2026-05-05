"""
Scores EV-level simulation metrics per snapshot bucket.

Per-bucket metrics
path_deviation  : time-bucket weighted penalty for route deviation
delta_arrival   : time-bucket weighted penalty for late arrival
ev_wait_time    : Gaussian decay score for queue wait time
missed_deadline : proportion of non-direct-drive arrivals that missed

Aggregation across snapshots is the responsibility of SimulationScore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from helpers.io_helpers import infer_snapshot_interval_ms
from helpers.constants import PERCENTILES

PATH_DEVIATION_BUCKETS: list[tuple[float, int]] = [
    (5,            0),
    (10,           0),
    (15,           2),
    (30,           6),
    (60,           12),
    (float("inf"), 15),
]
PATH_DEVIATION_BUCKET_LABELS: list[str] = ["5", "10", "15", "30", "60", "60+"]

DELTA_ARRIVAL_BUCKETS: list[tuple[float, int]] = [
    (0,            0),
    (5,            1),
    (10,           2),
    (15,           3),
    (30,           6),
    (60,           10),
    (float("inf"), 15),
]
DELTA_ARRIVAL_BUCKET_LABELS: list[str] = ["0", "5", "10", "15", "30", "60", "60+"]

PERCENTILE_NAMES: list[str] = ["p25", "p50", "p75", "p90", "p95", "p99"]

EV_METRIC_WEIGHTS = {
    "path_deviation": 1,
    "delta_arrival": 1,
    "ev_wait_time": 3,
    "missed_deadline": 2,
}

WAIT_DECAY_MINUTES: float = 45.0


def bucket_score(col: str, buckets: list[tuple[float, int]]) -> pl.Expr:
    """
    Builds a Polars expression that computes the normalised bucket-penalty score
    for col within a group.

    Each row is assigned the weight of the bucket it falls into. The group score
    is mean(row_weight) / total_weight, so the score ranges between 0 and 1.
    """
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

    return (1.0 - weight_expr.sum() / pl.col(col).count()).alias(f"{col}_score")


def wait_score(col: str) -> pl.Expr:
    """
    Gaussian decay score for wait time: exp(-(x/WAIT_DECAY_MINUTES)²), averaged per group.

    A wait of 0 minutes scores 1.0; longer waits score progressively lower,
    with 45 minutes scoring roughly 0.37, and waits over 90 minutes scoring near 0.

    exp( -(45 / 45)² ) = exp(-1) ≈ 0.3679
    exp( -(90 / 45)² ) = exp(-4) ≈ 0.0183
    """
    return ((-((pl.col(col) / WAIT_DECAY_MINUTES) ** 2)).exp())


def missed_deadline_score() -> list[pl.Expr]:
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
    Computes per-snapshot EV scores for a simulation run.

    Reads arrival_snapshots.parquet and waittime_snapshots.parquet, aligns
    them onto a shared bucket grid, and returns an EVScores dataclass.

    Aggregation across snapshots is the responsibility of SimulationScore.

    Args:
        run_id: Simulation run identifier (e.g. 'Run_001').
        output_root: Root directory containing all run output.

    Returns:
        EVScores with per-snapshot DataFrame only.
    """
    base_dir = output_root / run_id

    arrivals = pl.read_parquet(base_dir / "analysis" / "arrival_snapshots.parquet")

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
            bucket_score("path_deviation_minutes", PATH_DEVIATION_BUCKETS),
            bucket_score("delta_arrival_minutes",  DELTA_ARRIVAL_BUCKETS),
            *missed_deadline_score(),
        ])
    )

    wait_scores = (
        pl.read_parquet(base_dir / "percentiles" / "waittime" / "waittime_percentiles.parquet")
        .with_columns([
            *[wait_score(f"wait_p{name[1:]}").alias(f"wait_score_{name}") for name in PERCENTILE_NAMES]
        ]).with_columns([
            pl.mean_horizontal([f"wait_score_{name}" for name in PERCENTILE_NAMES])
              .alias("ev_wait_time_score")
        ])
    )

    quantile_cols = [f"wait_{name}" for name in PERCENTILE_NAMES]

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

    return EVScores(per_snapshot=per_snapshot)


@dataclass
class EVScores:
    per_snapshot: pl.DataFrame