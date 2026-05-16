"""
Scores station-level simulation metrics per snapshot bucket.

Per-bucket metrics
utilization_score : mean observed utilization across stations
expected_wait_score : mean Gaussian decay score for expected wait time
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from analysis.scoring.default_scores import (
    EXPECTED_WAIT_TIME_BUCKETS,
    PERCENTILE_NAMES,
    TOTAL_WAIT_WEIGHT,
)

WAIT_DECAY_MINUTES: float = 45.0


def expected_wait_score(wait_time_column: str) -> pl.Expr:
    """
    Gaussian decay score for wait time: exp(-(x/WAIT_DECAY_MINUTES)²).

    A wait of 0 minutes scores 1.0; longer waits score progressively lower,
    with 45 minutes scoring roughly 0.37, and waits over 90 minutes scoring near 0.

    exp( -(45 / 45)² ) = exp(-1) ≈ 0.3679
    exp( -(90 / 45)² ) = exp(-4) ≈ 0.0183
    """
    return (-((pl.col(wait_time_column) / WAIT_DECAY_MINUTES) ** 2)).exp()


def utilization_score(utilization_column: str) -> pl.Expr:
    return pl.col(utilization_column).clip(0.0, 1.0)


def compute_station_scores(
    run_id: str,
    output_root: Path,
    *,
    expected_wait_time_buckets: list[tuple[str, int]] = EXPECTED_WAIT_TIME_BUCKETS,
) -> StationScores:
    snapshots_path = output_root / run_id / "percentiles" / "station" / "station_percentiles.parquet"
    snapshots = pl.read_parquet(snapshots_path)

    per_bucket = snapshots.with_columns([
        *[utilization_score(f"utilization_{name}").alias(f"utilization_score_{name}") for name in PERCENTILE_NAMES]
    ]).with_columns([
        pl.mean_horizontal([f"utilization_score_{name}" for name in PERCENTILE_NAMES])
          .alias("utilization_score")
    ]).with_columns([
        *[expected_wait_score(f"wait_time_{name}").alias(f"wait_score_{name}") for name in PERCENTILE_NAMES]
    ]).with_columns([
        pl.sum_horizontal([
            pl.col(f"wait_score_{name}") * weight
            for name, weight in expected_wait_time_buckets
        ]).alias("expected_wait_score") / TOTAL_WAIT_WEIGHT
    ])

    return StationScores(
        per_bucket=per_bucket,
    )


@dataclass
class StationScores:
    per_bucket: pl.DataFrame