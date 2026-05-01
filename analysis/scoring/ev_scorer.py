"""
ev_scorer.py
Scores EV-level simulation metrics per snapshot tick:
  - path_deviation   (time-bucket weighted, per tick)
  - delta_arrival    (time-bucket weighted, per tick)
  - ev_wait_time     (Gaussian decay averaged over EVs at that tick)
  - missed_deadline  (proportion of non-direct arrivals that missed, per tick)

The run-wide aggregate for each metric is the mean of its per-tick scores.
The weighted_aggregate is the weighted mean of the four metric aggregates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

PATH_DEVIATION_BUCKETS: list[tuple[float, int]] = [
    (5, 1),
    (10, 1),
    (15, 1),
    (30, 1),
    (60, 2),
    (float("inf"), 3),
]
PATH_DEVIATION_BUCKET_LABELS = ["5", "10", "15", "30", "60", "60+"]

DELTA_ARRIVAL_BUCKETS: list[tuple[float, int]] = [
    (0, 2),
    (5, 1),
    (10, 1),
    (15, 1),
    (30, 1),
    (60, 1),
    (float("inf"), 1),
]
DELTA_ARRIVAL_BUCKET_LABELS = ["0", "5", "10", "15", "30", "60", "60+"]

METRIC_WEIGHTS: dict[str, int] = {
    "path_deviation": 1,
    "delta_arrival": 1,
    "ev_wait_time": 3,
    "missed_deadline": 2,
}


def bucket_score_expr(col: str, buckets: list[tuple[float, int]]) -> pl.Expr:
    """
    Builds a Polars expression that computes the bucket score for one column,
    grouped by simtime_ms. Returns a float per group.
    """
    total_weight = sum(w for _, w in buckets)

    # Build a weighted-penalty expression using pl.when/then chaining.
    # Each row gets the weight of whichever bucket it falls into.
    previous = float("-inf")
    weight_expr = pl.lit(None, dtype=pl.Float64)
    for upper, w in buckets:
        if upper == 0:
            cond = pl.col(col) == 0.0
        elif math.isinf(upper):
            cond = pl.col(col) > previous
        else:
            cond = (pl.col(col) > previous) & (pl.col(col) <= upper)
        weight_expr = pl.when(cond).then(pl.lit(float(w))).otherwise(weight_expr)
        previous = upper

    # score = mean(row_weight) / total_weight  (nulls are dropped by mean)
    return (weight_expr.mean() / total_weight).alias(f"{col}_score")


def wait_score_expr() -> pl.Expr:
    """Gaussian decay: exp(-(x/45)^2), averaged per group."""
    x = pl.col("wait_minutes")
    return ((-((x / 45.0) ** 2)).exp()).mean().alias("ev_wait_time_score")


def missed_deadline_expr() -> list[pl.Expr]:
    """Returns (proportion_missed, score = 1 - proportion) per group."""
    total  = pl.len()
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
    base = output_root / run_id / "analysis"

    arrivals = pl.read_parquet(base / "arrival_snapshots.parquet")

    wait_time_df = pl.read_parquet(base / "waittime_snapshots.parquet")
    
    snapshot_interval_ms = (
        wait_time_df.select("simtime_ms")
        .unique()
        .sort("simtime_ms")
        .with_columns(pl.col("simtime_ms").diff().alias("diff"))["diff"]
        .drop_nulls()
        .min()
    ) or 1

    arrival_scores = (
        arrivals
        .with_columns([
            ((pl.col("simtime_ms") // snapshot_interval_ms) * snapshot_interval_ms).alias("simtime_ms")
        ])
        .group_by("simtime_ms").agg([
            bucket_score_expr("path_deviation_minutes", PATH_DEVIATION_BUCKETS),
            bucket_score_expr("delta_arrival_minutes",  DELTA_ARRIVAL_BUCKETS),
            *missed_deadline_expr(),
        ])
    )

    wait_scores = wait_time_df.group_by("simtime_ms").agg([
        wait_score_expr(),
    ])

    per_snapshot = (
        arrival_scores
        .join(wait_scores, on="simtime_ms", how="left")
        .with_columns(pl.col("ev_wait_time_score").fill_null(1.0)) 
        .sort("simtime_ms")
        .rename({
            "path_deviation_minutes_score": "path_deviation_score",
            "delta_arrival_minutes_score": "delta_arrival_score",
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
        ])
    )

    path_deviation_agg = per_snapshot["path_deviation_score"].mean() or 0.0
    delta_arrival_agg = per_snapshot["delta_arrival_score"].mean() or 0.0
    wait_time_agg = per_snapshot["ev_wait_time_score"].mean() or 0.0
    missed_deadline_agg = per_snapshot["missed_deadline_score"].mean() or 0.0

    weighted_aggregate = (
        METRIC_WEIGHTS["path_deviation"] * path_deviation_agg
        + METRIC_WEIGHTS["delta_arrival"] * delta_arrival_agg
        + METRIC_WEIGHTS["ev_wait_time"] * wait_time_agg
        + METRIC_WEIGHTS["missed_deadline"] * missed_deadline_agg
    ) / sum(METRIC_WEIGHTS.values())

    print(f"Arrival keys: {arrival_scores['simtime_ms'].head(5).to_list()}")
    print(f"Wait keys: {wait_scores['simtime_ms'].head(5).to_list()}")
    print(f"Joined Null Count: {per_snapshot['ev_wait_time_score'].null_count()}")

    return EVScores(
        per_snapshot=per_snapshot,
        path_deviation_aggregate=path_deviation_agg,
        delta_arrival_aggregate=delta_arrival_agg,
        ev_wait_time_aggregate=wait_time_agg,
        missed_deadline_aggregate=missed_deadline_agg,
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
                    "bucket_labels":    PATH_DEVIATION_BUCKET_LABELS,
                    "bucket_weights":   [weights for _, weights in PATH_DEVIATION_BUCKETS],
                    "aggregate_score":  round(self.path_deviation_aggregate, 6),
                },
                "delta_arrival_minutes": {
                    "higher_is_better": False,
                    "bucket_labels":    DELTA_ARRIVAL_BUCKET_LABELS,
                    "bucket_weights":   [weights for _, weights in DELTA_ARRIVAL_BUCKETS],
                    "aggregate_score":  round(self.delta_arrival_aggregate, 6),
                },
                "ev_wait_time": {
                    "higher_is_better": False,
                    "aggregate_score":  round(self.ev_wait_time_aggregate, 6),
                },
                "missed_deadline": {
                    "higher_is_better": False,
                    "aggregate_score":  round(self.missed_deadline_aggregate, 6),
                },
            },
            "metric_weights":     METRIC_WEIGHTS,
            "weighted_aggregate": round(self.weighted_aggregate, 6),
        }