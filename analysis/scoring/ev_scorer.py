from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from analysis.scoring.scoring_config import ScoringConfig
from helpers.io_helpers import infer_snapshot_interval_ms

PERCENTILE_NAMES: list[str] = ["p25", "p50", "p75", "p90", "p95", "p99"]

def bucket_score(col: str, buckets: list[tuple[float, int]]) -> pl.Expr:
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


def wait_score(col: str, wait_decay_minutes: float) -> pl.Expr:
    return ((-((pl.col(col) / wait_decay_minutes) ** 2)).exp())


def missed_deadline_score() -> list[pl.Expr]:
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


def compute_ev_scores(run_id: str, output_root: Path, config: ScoringConfig) -> EVScores:
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
            bucket_score("path_deviation_minutes", config.path_deviation_buckets),
            bucket_score("delta_arrival_minutes",  config.delta_arrival_buckets),
            *missed_deadline_score(),
        ])
    )

    wait_scores = (
        pl.read_parquet(base_dir / "percentiles" / "waittime" / "waittime_percentiles.parquet")
        .with_columns([
            *[wait_score(f"wait_p{name[1:]}", config.wait_decay_minutes).alias(f"wait_score_{name}") for name in PERCENTILE_NAMES]
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
